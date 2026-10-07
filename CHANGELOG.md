# PhotonTracer Changelog

All notable changes to this project will be documented in this file.

## 1.0.x - Unreleased

- Add Changelog file
- Add CI workflow that compiles the OptiX build without a GPU
- Run the CI checks for PRs and pushes to the dev branch
- Add `benchmarks/benchmark.py` for timing synthetic scenes and comparing against a stored baseline
- Fix crash when a `Simulation` is used after another one has been destroyed (the destructor reset the whole CUDA device)
- Reject more than 16 materials, instance material IDs outside the configured materials, and mesh scenes with fewer than 2 materials; before, these silently absorbed rays or wrote past a fixed-size array
- Fix the debug-build bounds checks, which rejected the valid material ID 15
- Compact mesh acceleration structures after building, controlled by the new `compact` argument of `MeshGeometry` (default on)

## 1.0.2 - 18-03-2026

- Update citation information (ISSN and volume of the journal article)

## 1.0.1 - 16-03-2026

- Add citation information and update tests to match API changes

## 1.0.0 - 12-03-2026

- Initial release of PhotonTracer
