# Couplet — reference implementation

This is the reference implementation of **Couplet**, a dealer-free, actively secure two-party function secret sharing stack for private inference between two Web services.
The two parties generate every key and every piece of correlated randomness themselves by evaluating the dealer's generation circuits inside an actively secure Boolean two-party computation; evaluation yields MAC-authenticated sharings, and each nonlinear layer is a local read of one authenticated one-hot plane.

The package contains only source code. No build outputs, measurement records, logs or generated circuit binaries are shipped; all of them can be rebuilt from the files below.

## Layout

| path | content |
|---|---|
| `python/couplet/` | Python reference implementation (package `couplet`) |
| `python/tests/` | 17 Python test suites |
| `python/bench/` | campaign runner, table/figure generators, circuit exporter/regenerator |
| `python/vectors/` | test vectors shared by both languages |
| `cpp/include/couplet/`, `cpp/src/` | message-compatible C++ implementation (header-only library + 19 programs) |
| `cpp/tests/` | mixed-language interop tests (one party in C++, one in Python) and their data |
| `cpp/build.sh`, `cpp/build_msvc.bat` | per-target build scripts (called by `scripts/`) |
| `circuits/manifest.json` | SHA-256 digests of every generation circuit; the binaries are rebuilt from it |
| `scripts/` | one command per step (Linux and Windows) |
| `packaging/` | helper that assembles a results-bearing release package (optional) |

## Requirements

- An x86-64 CPU with AES-NI, PCLMULQDQ and AVX2.
- **Linux:** Ubuntu 22.04, g++ 11, Python 3.10 (a `venv` is created by the setup script).
- **Windows:** Visual Studio 2019 or newer with the C++ x64 tools; Python 3.10 or newer.
- Python packages: NumPy and matplotlib, pinned per interpreter in `requirements.txt`. No other library is used — only the standard library, sockets, threads and the OS random generator.

## Build, test and reproduce

| step | Linux | Windows |
|---|---|---|
| 1. install Python packages | `sh scripts/setup_python_linux.sh` | `python -m pip install -r requirements.txt` |
| 2. build the C++ programs | `sh scripts/build_linux.sh` | `scripts\build_windows.bat` |
| 3. generate the test circuits | `sh scripts/regen_circuits.sh` | `scripts\regen_circuits.bat` |
| 4. run all tests | `sh scripts/run_tests.sh` | `scripts\run_tests.bat` |
| 5. run a quick end-to-end campaign | `sh scripts/run_experiments.sh --quick` | `scripts\run_experiments.bat --quick` |
| 6. generate tables and figures | `sh scripts/make_tables.sh quick` | `scripts\make_tables.bat quick` |

Notes:

- **Step 3** rebuilds the circuits the tests load from `circuits/manifest.json` and verifies their SHA-256 digests. `regen_circuits.sh --missing` rebuilds all circuits listed in the manifest; the compiler is deterministic and the files are not shipped.
- **Step 4** runs the Python suites, the C++ test programs and the mixed-language tests; each program's log is written to `test_logs/` (created at run time).
- **Step 5** without `--quick` runs the full measurement campaign into `experiments/raw/campaign/`; `--quick` runs a short smoke configuration into `experiments/quick/campaign/`. All data are seeded and synthetic; no public dataset is used.
- **Step 6** regenerates the LaTeX table bodies, `numbers.tex`, `PROVENANCE.md` and the PDF figures from a campaign summary into `out/`; pass `campaign` instead of `quick` after a full run.

## License

MIT, see `LICENSE`.
