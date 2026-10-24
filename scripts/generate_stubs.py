"""Generate photontracer/photontracer_bindings.pyi from the docstrings and signatures of the bindings.

Run it with the Python that has the built package installed, after every change of the bindings:

    pip install pybind11-stubgen==3.0.0
    python scripts/generate_stubs.py           # writes the stub
    python scripts/generate_stubs.py --check   # fails if the committed stub is out of date
"""
import argparse
import difflib
import pathlib
import subprocess
import sys
import tempfile

STUB = pathlib.Path(__file__).resolve().parent.parent / "photontracer" / "photontracer_bindings.pyi"


def generate() -> str:
    with tempfile.TemporaryDirectory() as output:
        # Run from the empty output directory, so that the installed package is imported and not the
        # sources in this repository, which have no compiled module
        subprocess.run([sys.executable, "-m", "pybind11_stubgen", "--exit-code", "-o", output,
                        "photontracer.photontracer_bindings"], cwd=output, check=True)
        return (pathlib.Path(output) / "photontracer" / "photontracer_bindings.pyi").read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only compare the generated stub with the committed one")
    args = parser.parse_args()

    stub = generate()
    if not args.check:
        STUB.write_text(stub, encoding="utf-8")
        return 0

    committed = STUB.read_text(encoding="utf-8")
    if stub == committed:
        return 0
    sys.stdout.writelines(difflib.unified_diff(committed.splitlines(keepends=True), stub.splitlines(keepends=True),
                                               "committed", "generated"))
    print(f"\n{STUB.name} is out of date: run python scripts/generate_stubs.py and commit the result")
    return 1


if __name__ == "__main__":
    sys.exit(main())
