#!/usr/bin/env python
# This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
# © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg
"""Checks that two builds produce identical output.

Runs small fixed-seed simulations and hashes every output array. Run it on a
reference build and on the build under test, then compare:

    python benchmarks/golden.py --output reference.json
    python benchmarks/golden.py --compare reference.json

If scenes differ, save the arrays of both builds and diff them:

    python benchmarks/golden.py --arrays arrays_a        (reference build)
    python benchmarks/golden.py --arrays arrays_b        (build under test)
    python benchmarks/golden.py --diff arrays_a arrays_b

If the builds use different random number generators or backends, the arrays cannot be
identical. Compare their statistics instead: the mean of every output, as a z-score.

    python benchmarks/golden.py --rays 1000000 --summary summary_a.json     (reference build)
    python benchmarks/golden.py --rays 1000000 --summary summary_b.json     (build under test)
    python benchmarks/golden.py --compare-summary summary_a.json summary_b.json

Compare only builds made with the same compiler flags on the same GPU. With
--use_fast_math, nvcc may fuse multiplications and additions differently when
code moves between functions, which changes the last bits of float results.
Building both with CMAKE_CUDA_FLAGS=--fmad=false removes that effect.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import trimesh

import photontracer
from photontracer import (CameraRayGenerator, InstanceGeometry, IsotropicRayGenerator, Material, MaterialType,
                          MeshGeometry, OutputType, ParallelRayGenerator, Simulation)

SEED = 7
PRECISION = 1e-6  # relative floor when comparing means (about ten float ulps)
OUTPUTS = [
    OutputType.LAST_DIRECTION, OutputType.LAST_POSITION, OutputType.RAY_STATE, OutputType.SCATTERING_COUNT,
    OutputType.STOKES_VECTOR, OutputType.OPTICAL_PATH_LENGTH, OutputType.NUMBER_OF_WARNINGS, OutputType.LAST_MEDIUM_ID,
    OutputType.SOURCE_POSITION, OutputType.SOURCE_DIRECTION, OutputType.Q_MINUS_AXIS_IN, OutputType.STOKES_VECTOR_IN,
    OutputType.SCATTERING_ANGLE,
]
VACUUM = Material(MaterialType.REFRACTIVE, 1 + 0j)


def glass(n, k):
    return [VACUUM, Material(MaterialType.REFRACTIVE, complex(n, k))]


def sphere():
    s = trimesh.creation.icosphere(subdivisions=3, radius=5)
    return MeshGeometry(s.vertices, s.faces)


def particles(n=5):
    s = trimesh.creation.icosphere(subdivisions=2, radius=1)
    grid = np.stack(np.meshgrid(*[np.arange(n) * 2.2] * 3, indexing="ij"), -1).reshape(-1, 3) - n * 1.1
    transforms = np.tile(np.eye(4)[:3], (len(grid), 1, 1))
    transforms[:, :, 3] = grid
    ids = np.zeros(len(grid), dtype=np.uint32)
    return InstanceGeometry([MeshGeometry(s.vertices, s.faces)], transforms.astype(np.float32), ids, ids + 1)


def parallel(rays):
    return ParallelRayGenerator(number_of_rays=rays, origin=(0, 0, 14), direction=(0, 0, -1), offset_radius=5)


def isotropic(rays):
    return IsotropicRayGenerator(number_of_rays=rays, center=(0, 0, 0), source_radius=30, offset_radius=8)


def camera(rays):
    cam = CameraRayGenerator()
    cam.image_width, cam.image_height, cam.samples_per_pixel = 40, 30, 4
    cam.vertical_fov, cam.look_from, cam.look_at = 40.0, [0, 0, 20], [0, 0, 0]
    cam.vertical_up, cam.focus_distance, cam.defocus_angle = [0, 1, 0], 20.0, 2.0
    return cam


RANDOM_LINEAR = (1, np.nan, np.nan, 0)
RANDOM_CIRCULAR = (1, 0, 0, np.nan)

# name: geometry, materials, ray source, and optional settings
SCENES = {
    "glass_absorbing": dict(geometry=sphere, materials=glass(1.5, 2e-4)),
    "glass_real_fresnel": dict(geometry=sphere, materials=glass(1.5, 2e-4), complex_fresnel=False),
    "glass_random_linear": dict(geometry=sphere, materials=glass(1.5, 1e-3), stokes=RANDOM_LINEAR),
    "glass_random_circular": dict(geometry=sphere, materials=glass(1.31, 1e-4), stokes=RANDOM_CIRCULAR),
    "volume_scattering": dict(geometry=sphere, materials=[VACUUM, Material(MaterialType.VOLUME_SCATTERING, 1.33 + 1e-4j, 2.0, 0.8)]),
    "volume_scattering_isotropic_phase": dict(geometry=sphere, materials=[VACUUM, Material(MaterialType.VOLUME_SCATTERING, 1.33 + 1e-4j, 3.0, 0.0)], source=isotropic),
    "diffuse": dict(geometry=sphere, materials=[VACUUM, Material(MaterialType.DIFFUSE, 0.7)]),
    "reflective_fuzzy": dict(geometry=sphere, materials=[VACUUM, Material(MaterialType.REFLECTIVE, 0.9, 0.3)]),
    "reflective_mirror": dict(geometry=sphere, materials=[VACUUM, Material(MaterialType.REFLECTIVE, 1.0, 0.0)]),
    "particles_parallel": dict(geometry=particles, materials=glass(1.31, 1e-5)),
    "particles_isotropic": dict(geometry=particles, materials=glass(1.31, 1e-5), source=isotropic),
    "camera": dict(geometry=sphere, materials=glass(1.5, 2e-4), source=camera),
    "camera_random_linear": dict(geometry=sphere, materials=glass(1.5, 2e-4), source=camera, stokes=RANDOM_LINEAR),
    "healpix_q_seed": dict(geometry=sphere, materials=glass(1.31, 1e-4), source=isotropic, healpix=4, q_seed=(0, 1, 0), stokes=RANDOM_LINEAR),
    "q_seed_along_ray": dict(geometry=sphere, materials=glass(1.5, 1e-3), q_seed=(0, 0, 1)),
    "healpix_diffuse": dict(geometry=sphere, materials=[VACUUM, Material(MaterialType.DIFFUSE, 0.6)], healpix=2),
}


def run_scene(scene, rays, generator):
    sim = Simulation(gpu_id=0)
    if generator:
        sim.random_number_generator = getattr(photontracer.RandomNumberGenerator, generator)
    sim.geometry = scene["geometry"]()
    sim.materials = scene["materials"]
    sim.wavelength_um = 1.0
    sim.stokes_vector = scene.get("stokes", (1, 0, 0, 0))
    sim.seed = SEED
    sim.max_scattering_count = 5000
    sim.use_complex_fresnel = scene.get("complex_fresnel", True)
    sim.ray_generator = scene.get("source", parallel)(rays)
    if "q_seed" in scene:
        sim.q_minus_axis_seed = scene["q_seed"]
    outputs = list(OUTPUTS)
    if scene.get("healpix"):
        sim.direction_healpix_nside = scene["healpix"]
        outputs.append(OutputType.DIRECTION_HISTOGRAM_HEALPIX)
    sim.outputs = outputs
    sim.run()
    arrays = {str(o).split(".")[-1]: np.ascontiguousarray(sim.get_output_buffer(o)) for o in outputs}
    return arrays, generator_of(sim)


def digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(arrays[name].tobytes())
    return h.hexdigest()


def generator_of(sim):
    """Name of the generator of a simulation; builds before 1.1 only have curand."""
    return sim.random_number_generator.name if hasattr(sim, "random_number_generator") else "MRG32K3A"


def summarize(arrays):
    """Mean, variance and sample size per component of every output."""
    summary = {}
    for name, array in arrays.items():
        if name == "DIRECTION_HISTOGRAM_HEALPIX":
            summary[name] = {"counts": array.astype(np.float64).tolist()}
            continue
        values = array.reshape(len(array), -1).astype(np.float64)
        summary[name] = {"mean": np.nanmean(values, axis=0).tolist(), "var": np.nanvar(values, axis=0).tolist(),
                         "n": np.sum(~np.isnan(values), axis=0).tolist()}
    return summary


def z_scores(a, b):
    """Differences of the means in standard errors, as (name, component, z) tuples."""
    scores = []
    for name in a:
        if name not in b:
            continue
        if "counts" in a[name]:  # Poisson counts per pixel
            ca, cb = np.array(a[name]["counts"]), np.array(b[name]["counts"])
            ca, cb = ca / ca.sum(), cb / cb.sum()
            na, nb = np.sum(a[name]["counts"]), np.sum(b[name]["counts"])
            sigma = np.sqrt(ca / na + cb / nb)
            for pixel in np.nonzero(sigma)[0]:
                scores.append((name, int(pixel), float((ca[pixel] - cb[pixel]) / sigma[pixel])))
            continue
        for c, (ma, va, na) in enumerate(zip(a[name]["mean"], a[name]["var"], a[name]["n"])):
            mb, vb, nb = b[name]["mean"][c], b[name]["var"][c], b[name]["n"][c]
            # Float rounding can shift a mean by a few ulps even if its spread is tiny (the intensity
            # of the Stokes vector is 1 up to rounding), so add a relative precision floor
            precision = PRECISION * max(abs(ma), abs(mb), np.sqrt(max(va, vb)))
            sigma = np.sqrt(va / na + vb / nb + precision**2)
            if sigma > 0:
                scores.append((name, c, float((ma - mb) / sigma)))
            elif ma != mb:
                scores.append((name, c, float("inf")))
    return scores


def compare_summaries(path_a, path_b, limit=5.0):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    print(f"A: {a.get('label') or '?'} ({a.get('rng')}), B: {b.get('label') or '?'} ({b.get('rng')})")
    ok = True
    total = 0
    for scene in a["scenes"]:
        if scene not in b["scenes"]:
            continue
        scores = z_scores(a["scenes"][scene], b["scenes"][scene])
        total += len(scores)
        worst = max(scores, key=lambda t: abs(t[2]), default=("-", 0, 0.0))
        flag = "  TOO DIFFERENT" if abs(worst[2]) > limit else ""
        ok &= abs(worst[2]) <= limit
        print(f"  {scene:36s} {len(scores):4d} values, largest |z| = {abs(worst[2]):5.2f} ({worst[0]} {worst[1]}){flag}")
    print(f"{total} comparisons; by chance about {total * 5.7e-7:.4f} of them would exceed |z| > {limit}")
    return ok


def compare_hashes(results, reference):
    ok = True
    print(f"\nCompared with {reference.get('label', '?')}")
    for name, value in results.items():
        expected = reference["scenes"].get(name)
        status = "no reference" if expected is None else ("same" if expected == value else "DIFFERENT")
        ok &= status != "DIFFERENT"
        print(f"  {name:36s} {status}")
    return ok


def diff_arrays(dir_a, dir_b):
    for file in sorted(os.listdir(dir_a)):
        if not file.endswith(".npz") or not os.path.exists(os.path.join(dir_b, file)):
            continue
        a, b = np.load(os.path.join(dir_a, file)), np.load(os.path.join(dir_b, file))
        print(file[:-4])
        for key in a.files:
            x, y = a[key], b[key]
            differs = (x != y) & ~(np.isnan(x) & np.isnan(y))
            count = int(differs.sum())
            largest = float(np.max(np.abs(x[differs].astype(np.float64) - y[differs]))) if count else 0.0
            print(f"  {key:28s} {count:9d} of {x.size:9d} differ, largest difference {largest:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", action="append", choices=SCENES, help="scene to run (default: all)")
    parser.add_argument("--rng", choices=["PCG32", "MRG32K3A"], help="random number generator (default: that of the build)")
    parser.add_argument("--rays", type=int, default=200_000, help="rays per scene")
    parser.add_argument("--label", help="name of this build in the results")
    parser.add_argument("--output", help="write the hashes to this JSON file")
    parser.add_argument("--compare", help="JSON file of a reference build to compare with")
    parser.add_argument("--arrays", help="save the output arrays of every scene in this directory")
    parser.add_argument("--summary", help="write the statistics of every output to this JSON file")
    parser.add_argument("--compare-summary", nargs=2, metavar=("A", "B"), help="compare two --summary files and exit")
    parser.add_argument("--diff", nargs=2, metavar=("DIR_A", "DIR_B"), help="compare two --arrays directories and exit")
    args = parser.parse_args()

    if args.diff:
        diff_arrays(*args.diff)
        return
    if args.compare_summary:
        sys.exit(0 if compare_summaries(*args.compare_summary) else 1)

    if args.arrays:
        os.makedirs(args.arrays, exist_ok=True)
    results = {}
    summaries = {}
    generator = None
    for name in args.scene or SCENES:
        arrays, generator = run_scene(SCENES[name], args.rays, args.rng)
        results[name] = digest(arrays)
        if args.summary:
            summaries[name] = summarize(arrays)
        print(f"{name:36s} {results[name][:20]}", flush=True)
        if args.arrays:
            np.savez(os.path.join(args.arrays, name + ".npz"), **arrays)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"label": args.label or "", "rng": generator, "rays": args.rays, "scenes": results}, f, indent=2)
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump({"label": args.label or "", "rng": generator, "rays": args.rays, "scenes": summaries}, f)
    if args.compare:
        with open(args.compare) as f:
            reference = json.load(f)
        if reference.get("rng", "MRG32K3A") != generator:
            print(f"\nThe reference used the {reference.get('rng', 'MRG32K3A')} generator and this run uses "
                  f"{generator}, so the hashes cannot match. Compare statistics with --summary instead.")
            sys.exit(2)
        if not compare_hashes(results, reference):
            sys.exit(1)


if __name__ == "__main__":
    main()
