# Couplet: artifact

Couplet is a dealer-free, actively secure two-party function secret sharing stack for private
inference between two Web services. The two parties generate every key and every piece of
correlated randomness themselves, by evaluating the dealer's generation circuits inside an actively
secure Boolean two-party computation; evaluation yields MAC-authenticated sharings, and every
nonlinear layer of the inference is a local read of one authenticated one-hot plane. This package
holds the Python reference implementation, the message-compatible C++ implementation with its
two-process drivers, all tests, the measurement campaign, and the outputs reported in the paper.

## Layout

| path | content |
|---|---|
| `python/couplet/` | reference implementation (package `couplet`) |
| `python/tests/` | 17 test suites (plain scripts; the last five cover the inference layer) |
| `python/vectors/` | test vectors shared by both languages |
| `python/bench/` | campaign runner, table and figure generators, circuit exporter and regenerator |
| `cpp/include/couplet/`, `cpp/src/` | C++ implementation (header only) and its 19 programs |
| `cpp/tests/` | mixed-language tests (one party in C++, one in Python) and their data |
| `cpp/build.sh`, `cpp/build_msvc.bat` | per-target build scripts (called by `scripts/`) |
| `circuits/manifest.json` | digests of every generation circuit; the circuit files are rebuilt from it |
| `scripts/` | the commands below |
| `results/` | final outputs: campaign summary, machine description, attack records, tables, figures, test log |

## Requirements

Linux: Ubuntu 22.04, g++ 11, Python 3.10 with `venv`; Windows: Visual Studio 2019 or newer with the
C++ x64 tools, Python 3.10 or newer. Both: an x86-64 CPU with AES-NI, PCLMULQDQ and AVX2, and NumPy
and matplotlib as pinned in `requirements.txt`. No other library is used (sockets, threads and the
operating system's random generator only).

## Commands

| step | Linux | Windows |
|---|---|---|
| 1. Python packages | `sh scripts/setup_python_linux.sh` | `python -m pip install -r requirements.txt` |
| 2. build | `sh scripts/build_linux.sh` | `scripts\build_windows.bat` |
| 3. circuit files of the tests | `sh scripts/regen_circuits.sh` | `scripts\regen_circuits.bat` |
| 4. tests | `sh scripts/run_tests.sh` | `scripts\run_tests.bat` |
| 5. experiments | `sh scripts/run_experiments.sh --quick` | `scripts\run_experiments.bat --quick` |
| 6. tables and figures | `sh scripts/make_tables.sh` | `scripts\make_tables.bat` |

Step 3 rebuilds the 36 circuits that the tests load from `circuits/manifest.json` and checks their
SHA-256 digests (`--missing`: all 75, 3.6 GB). Step 4 runs the Python suites, the 9 C++ test
programs and the 5 mixed-language tests; logs go to `test_logs/`. Step 5 without `--quick` runs the
whole campaign into `experiments/raw/campaign/` (it first rebuilds all circuits); with `--quick` a
smoke run of every driver on small configurations into `experiments/quick/campaign/`. Step 6
regenerates `results/tables` and `results/figures` from `results/summary.json` into `out/` and
compares them (`make_tables quick` or `make_tables campaign` uses the summary of step 5 instead).

## Expected run times

Measured on the authors' machine (Intel Core i9-14900K, 192 GiB, Windows 11, MSVC 19.51,
CPython 3.14.7, NumPy 2.5.3): build 48 s; circuit files of the tests 39 s; tests 35 minutes in
total (Python suites 27 minutes, C++ programs 64 s, mixed-language tests 7 minutes); quick
experiments 1 to 2 minutes; tables and figures 2 s. The whole campaign took about 7.2 hours
of two-process runs (694 runs of 118 configurations, plus the adversarial regression, 20 minutes);
its largest rows need 36 GiB of memory per party (73 GiB for both processes).

## Results

`results/summary.json` (every configuration with its runs' statistics and checks), `summary.txt`
(the headline tables as text), `machine.json`, `attacks.json` and `attacks_ppml.json` (adversarial
regression), `tables/` (LaTeX table bodies, `numbers.tex` with every number of the paper as a
macro, `PROVENANCE.md` mapping each to its JSON field), `figures/` (PDF), and `test_log.txt` (the
complete output of step 4 on the packaged tree). All data are synthetic (seeded); no public
dataset is used.

## What was exercised

On the authors' machine (Windows 11, MSVC, Python 3.14.7 with NumPy 2.5.3), on this packaged tree:
steps 1 to 6 with the Windows scripts (`results/test_log.txt`: 31 programs, 31 passed), and the Linux scripts
`regen_circuits.sh`, `run_tests.sh`, `run_experiments.sh` and `make_tables.sh` under the POSIX
shell of Git for Windows. The Python suites also passed with NumPy 2.3.5 (Python 3.12.14). The
campaign in `results/` was run before packaging; packaging changed only `#include` lines of the C++
sources and options of the Python runner and generators (the tables regenerate byte for byte).

Not exercised: the Linux scripts were written for Ubuntu 22.04 (g++ 11, Python 3.10, NumPy 2.2.6,
matplotlib 3.10.9) but not executed there; the C++ code has not been compiled with g++. Instead it
was checked for portability: every MSVC-specific construct has a guarded POSIX/GCC branch; every
file includes the standard headers it uses; all programs compile under MSVC's strict conformance
mode (`/permissive-`); and the test programs pass with a 128-bit integer type that converts signed
values like `unsigned __int128` (the GCC type) does. The Python sources parse with the Python 3.10
grammar. On Linux the campaign runner records no background load and no peak memory (both are read
from Windows counters), and figures are byte-identical to `results/figures` only with matplotlib
3.11.2.

Source comments refer to sections of internal design notes that are not part of this package; the
paper covers the same material.

## License

MIT (see `LICENSE`).
