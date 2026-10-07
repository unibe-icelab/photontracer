# Benchmarks and regression checks

Both scripts need a built `photontracer` and a GPU.

## benchmark.py

Times three synthetic scenes (slab, sphere, a large cloud of instanced particles) and reports
the first run (includes the acceleration structure build), the median over repeats, Mrays/s,
GPU memory and a mean scattering count as a sanity check.

    python benchmarks/benchmark.py --label before --output benchmarks/results/before.json
    python benchmarks/benchmark.py --compare benchmarks/results/before.json --tolerance 0.1

`--compare` exits with 1 if a scene is slower than the tolerance or the scattering count changed.
`--rng PCG32|MRG32K3A` selects the random number generator. Timings depend on the machine, so compare only runs from the same GPU. When the baseline was made
with another random number generator, the scattering counts only have to agree to 1%.

## golden.py

Runs 16 small fixed-seed scenes and hashes every output array, to show that a refactoring did
not change the results. Use it like the benchmark: save the hashes of a reference build with
`--output`, then run the build under test with `--compare`.

`--rng PCG32|MRG32K3A` selects the random number generator; builds before 1.1 only have `MRG32K3A`, and the hashes of the two generators cannot match.

If scenes differ, `--arrays DIR` saves all output arrays and `--diff DIR_A DIR_B` lists which
outputs differ and by how much.

The hashes are only comparable between builds made with the same compiler flags. With
`--use_fast_math`, nvcc fuses multiplications and additions differently when code moves
between functions, so the last bits of float results can change, mostly in the Stokes vector
and in rays that are computed from several float operations (camera, isotropic source). For an
exact comparison, configure both builds with `-DCMAKE_CUDA_FLAGS=--fmad=false`.

Runs with a different random number generator (`Simulation.random_number_generator`, see the main README)
or a different backend cannot give identical arrays. Compare the mean of every output instead:
`--summary FILE` stores the means, variances and counts of all outputs, and
`--compare-summary A B` lists, per scene, the largest difference in standard errors (z-score).
Use 1 million rays or more. A relative floor of 1e-6 ignores float rounding in quantities that
are constant up to rounding, such as the intensity of the Stokes vector. The comparison finds
differences of a few percent in a mean at 1 million rays; smaller ones need more rays.
`--compare` refuses to compare hashes of different generators.

Results are written to `benchmarks/results/`, which is not tracked by git.
