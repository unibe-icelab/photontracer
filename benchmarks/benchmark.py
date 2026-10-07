#!/usr/bin/env python
# This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
# © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg
"""Timing benchmark on small synthetic scenes.

    python benchmarks/benchmark.py --output results.json
    python benchmarks/benchmark.py --compare baseline.json --tolerance 0.1

Compare timings only between runs on the same machine and GPU.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import trimesh

import photontracer
from photontracer import (InstanceGeometry, Material, MaterialType, MeshGeometry,
                          OutputType, ParallelRayGenerator, Simulation)

SEED = 1234
MATERIALS = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.5 + 1e-5j)]


def slab_scene():
    slab = trimesh.creation.box(extents=[10, 10, 1],
                                transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    k = np.array([-np.sin(np.pi / 4), 0, -np.cos(np.pi / 4)])
    return MeshGeometry(slab.vertices, slab.faces), ParallelRayGenerator, dict(origin=-3 * k, direction=k, offset_radius=4)


def sphere_scene():
    sphere = trimesh.creation.icosphere(subdivisions=6, radius=5)
    return MeshGeometry(sphere.vertices, sphere.faces), ParallelRayGenerator, dict(
        origin=(0, 0, 10), direction=(0, 0, -1), offset_radius=5)


def instances_scene(n=68):
    sphere = MeshGeometry(*(lambda m: (m.vertices, m.faces))(trimesh.creation.icosphere(subdivisions=3, radius=1)))
    rng = np.random.default_rng(SEED)
    grid = np.stack(np.meshgrid(*[np.arange(n) * 2.2] * 3, indexing="ij"), -1).reshape(-1, 3)
    grid = grid + rng.uniform(-0.1, 0.1, grid.shape) - n * 1.1
    transforms = np.tile(np.eye(4)[:3], (len(grid), 1, 1))
    transforms[:, :, 3] = grid
    ids = np.zeros(len(grid), dtype=np.uint32)
    geometry = InstanceGeometry([sphere], transforms.astype(np.float32), ids, np.ones_like(ids))
    return geometry, ParallelRayGenerator, dict(origin=(0, 0, n * 1.1 + 10), direction=(0, 0, -1), offset_radius=n * 1.1)


SCENES = {"slab": slab_scene, "sphere": sphere_scene, "instances": instances_scene}


def gpu_memory_mb():
    """Device memory used by this process, or None if nvidia-smi can't tell."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            pid, mem = (v.strip() for v in line.split(","))
            if int(pid) == os.getpid():
                return float(mem)
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return None


def run_scene(name, rays, repeats, grid, rng):
    geometry, generator, kwargs = instances_scene(grid) if name == "instances" else SCENES[name]()
    sim = Simulation(gpu_id=0)
    if rng:
        sim.random_number_generator = getattr(photontracer.RandomNumberGenerator, rng)
    sim.geometry = geometry
    sim.materials = MATERIALS
    sim.wavelength_um = 1.0
    sim.stokes_vector = [1, 0, 0, 0]
    sim.seed = SEED
    sim.max_scattering_count = 10000
    sim.ray_generator = generator(number_of_rays=rays, **kwargs)
    sim.outputs = [OutputType.SCATTERING_COUNT]

    # The first run also builds the acceleration structure and the pipeline.
    t0 = time.perf_counter()
    sim.run()
    first_ms = (time.perf_counter() - t0) * 1e3
    memory = gpu_memory_mb()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sim.run()
        times.append((time.perf_counter() - t0) * 1e3)
    median = statistics.median(times)
    return {
        "scene": name,
        "rays": rays,
        "first_run_ms": round(first_ms, 2),
        "median_ms": round(median, 2),
        "min_ms": round(min(times), 2),
        "mrays_per_s": round(rays / median / 1e3, 2),
        "gpu_memory_mb": memory,
        "mean_scattering_count": float(sim.get_output_buffer(OutputType.SCATTERING_COUNT).mean()),
        "rng": generator_of(sim),
    }


def generator_of(sim):
    """Name of the generator of a simulation; builds before 1.1 only have curand."""
    return sim.random_number_generator.name if hasattr(sim, "random_number_generator") else "MRG32K3A"


def environment(label, generator):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        commit, gpu = "", ""
    return {"label": label or commit, "rng": generator, "commit": commit, "gpu": gpu,
            "python": platform.python_version(), "platform": platform.platform()}


def compare(results, baseline, tolerance):
    """Print the change against a baseline file; return True if nothing regressed."""
    old = {r["scene"]: r for r in baseline["results"]}
    ok = True
    generator = results[0]["rng"]
    other_generator = baseline["environment"].get("rng", "MRG32K3A") != generator
    print(f"\nCompared with baseline {baseline['environment'].get('label', '?')} ({baseline['environment'].get('gpu', '?')})")
    if other_generator:
        print(f"The baseline used the {baseline['environment'].get('rng', 'MRG32K3A')} generator and this build uses "
              f"{generator}: the scattering counts only need to agree to 1%.")
    for r in results:
        b = old.get(r["scene"])
        if b is None:
            print(f"  {r['scene']:<10} no baseline")
            continue
        change = r["median_ms"] / b["median_ms"] - 1
        same = np.isclose(r["mean_scattering_count"], b["mean_scattering_count"], rtol=1e-2 if other_generator else 1e-6)
        flags = ("  SLOWER" if change > tolerance else "") + ("" if same else "  RESULT CHANGED")
        ok &= change <= tolerance and same
        print(f"  {r['scene']:<10} {b['median_ms']:9.1f} -> {r['median_ms']:9.1f} ms ({change:+.1%}){flags}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", action="append", choices=SCENES, help="scene to run (default: all)")
    parser.add_argument("--rng", choices=["PCG32", "MRG32K3A"], help="random number generator (default: that of the build)")
    parser.add_argument("--rays", type=int, default=10_000_000, help="rays per scene")
    parser.add_argument("--grid", type=int, default=68, help="instances scene: spheres per side (grid**3 instances)")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs after the first")
    parser.add_argument("--label", help="name for this run in the results (default: git commit)")
    parser.add_argument("--output", help="write the results to this JSON file")
    parser.add_argument("--compare", help="baseline JSON file to compare against")
    parser.add_argument("--tolerance", type=float, default=0.1, help="allowed slowdown (0.1 = 10%%)")
    args = parser.parse_args()

    results = []
    print(f"{'scene':<10} {'first [ms]':>10} {'median [ms]':>12} {'Mrays/s':>9} {'GPU [MB]':>9}  mean scatter")
    for name in args.scene or SCENES:
        r = run_scene(name, args.rays, args.repeats, args.grid, args.rng)
        results.append(r)
        print(f"{name:<10} {r['first_run_ms']:>10.1f} {r['median_ms']:>12.1f} {r['mrays_per_s']:>9.2f} "
              f"{r['gpu_memory_mb'] or float('nan'):>9.0f}  {r['mean_scattering_count']:.4f}")

    report = {"environment": environment(args.label, results[0]["rng"]), "results": results}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    if args.compare:
        with open(args.compare) as f:
            if not compare(results, json.load(f), args.tolerance):
                sys.exit(1)


if __name__ == "__main__":
    main()
