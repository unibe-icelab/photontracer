# Benchmarks and regression checks

Both scripts need a built `photontracer` and a GPU.

## benchmark.py

Times three synthetic scenes (slab, sphere, a large cloud of instanced particles) and reports
the first run (includes the acceleration structure build), the median over repeats, Mrays/s,
GPU memory and a mean scattering count as a sanity check.

    python benchmarks/benchmark.py --label before --output benchmarks/results/before.json
    python benchmarks/benchmark.py --compare benchmarks/results/before.json --tolerance 0.1

`--compare` exits with 1 if a scene is slower than the tolerance or the scattering count changed.
Timings depend on the machine, so compare only runs from the same GPU.

## golden.py

Runs 16 small fixed-seed scenes and hashes every output array, to show that a refactoring did
not change the results. Use it like the benchmark: save the hashes of a reference build with
`--output`, then run the build under test with `--compare`.

If scenes differ, `--arrays DIR` saves all output arrays and `--diff DIR_A DIR_B` lists which
outputs differ and by how much.

The hashes are only comparable between builds made with the same compiler flags. With
`--use_fast_math`, nvcc fuses multiplications and additions differently when code moves
between functions, so the last bits of float results can change, mostly in the Stokes vector
and in rays that are computed from several float operations (camera, isotropic source). For an
exact comparison, configure both builds with `-DCMAKE_CUDA_FLAGS=--fmad=false`.

The results of a different random number generator cannot be compared this way.

Results are written to `benchmarks/results/`, which is not tracked by git.
