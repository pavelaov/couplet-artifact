"""Measurement campaign of the Couplet evaluation (DESIGN_M3.md section 6 with the
corrections C7 and C8 of design/CHANGELOG_M3.md).

Runs every configuration with the native programs of ``artifact/cpp/build``,
checks every record and writes

    artifact/experiments/raw/machine.json             the machine (system queries)
    artifact/experiments/models/                      the trained models, their test vectors,
                                                      tables, plan files and accuracy reports
    artifact/experiments/raw/campaign/<name>.json     one file per two-process configuration:
                                                      every repetition with both parties' records
    artifact/experiments/raw/campaign/<name>_p0.json, _p1.json   the records of the median run
    artifact/experiments/raw/campaign/tools.json      results of the single-program groups
    artifact/experiments/raw/campaign/summary.json    the consolidated summary
    artifact/experiments/raw/campaign/summary.txt     the headline tables as text
    artifact/experiments/raw/bench_ppml_local.json    group l (bench_ppml)
    artifact/experiments/raw/baselines_cpp.json       group k (bench_baselines)
    artifact/experiments/raw/attacks.json, attacks_ppml.json   group a

Groups (``--only m,u`` selects; default: all, in this order):

    m  models: train every model on the seeded synthetic task
       (couplet.tools.train_synthetic), export it with 16 test inputs
       (couplet.tools.export_model), write its plan files and circuits
       (bench/export_circuits.py --plan) and an accuracy report: float accuracy,
       accuracy of the fixed-point mirror in both truncation modes and their
       prediction agreement with the float model on the whole synthetic test set
       (2,048 inputs).  No public dataset is used anywhere.
    u  unit costs of the correlated randomness (couplet_item, couplet_party): one
       site(n, f, mode) for (n, f) in {(8, 6), (12, 8), (14, 8), (16, 8), (20, 8)},
       both modes, batch 1 and 64; one Beaver triple and a batch of 64; one matrix
       triple 16 x 16; one input mask; one output mask; one DCF key with payload 1
       at the same n (C7: the baseline of the price inversion); the comparison
       bundle of the core (two DCF keys, wrap bits, authenticated offset) at n in
       {8, 16} as an extra row
    k  key generation: a DPF key with payload 1 (couplet_party) for n in {8, 12, 16,
       20} and the re-implemented one-bit-leaky and semi-honest protocols
       (bench_baselines) at the same n
    l  local plane work (bench_ppml): plane evaluation, prefix tables, table and
       locator reads for n in {8, 12, 14, 16, 20}, 1 and 8 threads
    e  end-to-end inference, public-model mode, no delay: MLPs [16,8,8,4],
       [32,16,16,8], [64,32,32,10], [784,128,128,10] (tanh), CNNs 10 x 10 and 14 x 14
       (reduced) and 28 x 28 (the full shape; chunked triple requests), both modes,
       formats (16, 8) and (14, 8), one thread; 8 threads for the two largest models
    o  online layer: the full MLP at (12, 8), and the MLPs [784, 128, 128, 128, 10]
       and [784, 128, 10] at (12, 8), (14, 8), (16, 8), both modes; the difference
       [784,128,128,128,10] - [784,128,128,10] is one authenticated hidden layer of
       128 sites (per-layer figures of the summary, online_layers)
    p  private-model mode (correction C2 output masks) for [16,8,8,4] and
       [32,16,16,8] at (16, 8), both modes; the full MLP [784,128,128,10] at (16, 8),
       both modes (PRIVATE_FULL: a tiled plan, more than 4,096 weights, task R-TILE;
       triple requests of at most CHUNK_AND_PRIVATE triples, so the name carries
       _c40000000); a configuration of PRIVATE_FULL without an accepted file enters
       the summary as not_run with an estimate from its plan file and the measured
       rows (private_full_assessment, written to private_full.json)
    w  emulated wide-area links at (16, 8): round-trip time 20 ms and 80 ms (one-way
       delay 10 / 40 ms) at 100 Mbit/s per direction (token bucket of
       couplet/netem.hpp in both drivers' channel), every model of group e; a
       configuration whose estimated run time (its no-delay row plus bytes / rate
       plus rounds times the delay) exceeds --wan-max-run seconds is recorded as
       not feasible with the estimate
    a  adversarial regression: python -m couplet.attacks.run_all (core rows) and
       bench/ppml_attacks.py (inference rows, C2 output-mask test)

Method of the two-process runs (that of the core's campaign runner).  Each
configuration is run ``--repeat`` times (default 9) if one run takes less than
ten seconds, 5 times up to a minute and 3 times above (the first run decides),
always with ``--verify``: two processes on 127.0.0.1, real preprocessing,
sigma = 40, one thread per party unless the configuration says otherwise.
The machine is shared with other jobs: for every run the runner measures the
processor time that the REST of the machine used while the run lasted (the
run's own process tree is read from a job object; percent of all logical
processors).  A run whose background load exceeds ``--quiet-percent`` (10, the
core's value; ``--quiet-percent-wan``, 30, for the rows on an emulated wide-area
link, whose time is dominated by the emulated transfers and delays; see
design/CHANGELOG_M3.md entry M3-E) stays in the file with ``"timing_valid":
false`` and does not enter the time statistics (the summary recomputes the flag
from the recorded load with the thresholds of its own invocation);
the runner repeats until it has the required valid runs (at most three times as
many attempts).  Before each run the runner waits (at most ``--quiet-wait``
seconds) for a half-second sample below the same percentage.  A configuration
is accepted only if, in every run, both processes exit with status 0, the
per-phase counters sum to the channel totals at both parties, the bytes sent by
one party equal the bytes received by the other, both count the same rounds,
the session drew its triples as planned (ONE request that covers exactly its
circuits, or the chunked requests of couplet_infer --chunk-and) in strict mode
with every triple used, the outputs verified, and bytes and rounds, in total
and per phase, are identical in all runs.  The inference runs use a different
test input in every repetition (input index = repetition mod 16); their logits
are compared with the driver's local check (mode exact: the exported exact
logits; mode round: the C++ mirror on the thresholds of the run) and again with
the Python mirror of couplet.nn.  The reported time of a run is the wall-clock
time of the slower party from its first protocol message to its last output;
the summary gives median, minimum and maximum over the valid repetitions.  The
peak memory (working set and commit charge of both processes, from the
operating system) is that of the timed runs; the TEST-ONLY verification after
the measured part is small against the preprocessing.  A configuration that
fails or is not feasible stays in the summary with its status and reason.

Usage (from ``artifact/python``; build ``artifact/cpp`` first, cpp/build_msvc.bat
or cpp/build.sh):

    python bench/run_campaign.py                  # everything not yet in the output directory
    python bench/run_campaign.py --only m,u,k,l   # some groups
    python bench/run_campaign.py --list
    python bench/run_campaign.py --force --match mlp16
    python bench/run_campaign.py --summary-only   # rebuild summary.json / summary.txt
    python bench/run_campaign.py --quick          # smoke run (minutes): output in experiments/quick

``--quick`` runs the groups m, u, k, l, e, w on a few small configurations only (quick_keep:
site(8, 6), a Beaver triple, an input and an output mask, DCF n = 8, DPF n = 8 and 16, the MLP
[16,8,8,4] at (16, 8) without delay and on the 20 ms link; bench_ppml --quick and bench_baselines
with two repetitions, at n = 8 and 16) with one repetition each, and writes everything that the full campaign
writes under experiments/raw into experiments/quick (the models stay in experiments/models).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

ARTIFACT = os.path.normpath(os.path.join(_HERE, "..", ".."))
CPP = os.path.join(ARTIFACT, "cpp")
CIRCUITS = os.path.join(ARTIFACT, "circuits")
RAW = os.path.join(ARTIFACT, "experiments", "raw")
MODELS_DIR = os.path.join(ARTIFACT, "experiments", "models")
DEFAULT_OUT = os.path.join(RAW, "campaign")
SCHEMA = "couplet.m3.campaign/1"

K = S = 64
SIGMA = 40
MODES = ("round", "exact")
FORMATS = ((16, 8), (14, 8))
ONLINE_FORMATS = ((12, 8), (14, 8), (16, 8))
UNIT_NF = ((8, 6), (12, 8), (14, 8), (16, 8), (20, 8))
KEYGEN_N = (8, 12, 16, 20)
E2E_MODELS = ("mlp16", "mlp32", "mlp64", "mlp784", "cnn10", "cnn14", "cnn28")
LARGEST = ("mlp784", "cnn28")
PRIVATE_MODELS = ("mlp16", "mlp32")
CHUNK_AND = {"cnn28": 40_000_000}          # triples per request (couplet_infer --chunk-and)
# Private-model mode of the full MLP (task R-PRIV): its plan is tiled (more than 4,096 weights, task R-TILE) and needs
# about 5.1 x 10^9 AND gates, so its triple requests are capped like those of the full CNN.  The cap applies to the
# private-model configurations of these presets only (the public mlp784 rows keep one request).
PRIVATE_FULL = ("mlp784",)
CHUNK_AND_PRIVATE = {"mlp784": 40_000_000}
N_INPUTS = 16                              # test inputs exported per model
LINKS = {"lan": (0.0, 0.0), "wan20": (10.0, 100.0), "wan80": (40.0, 100.0)}   # one-way delay (ms), Mbit/s
MODEL_SIZES = {"mlp784h1": [784, 128, 10], "mlp784h3": [784, 128, 128, 128, 10]}  # not presets of couplet.nn
TRAIN = {"epochs": 15, "train": 8192, "test": 2048, "data_seed": 20261013, "init_seed": 3, "input_seed": 7}

PREP_PHASES = ("baseot", "cotext", "abit", "laand", "bucket", "check")
ENGINE_PHASES = ("input", "online", "output")
INFER_PREP_GROUPS = ("base_ot", "ot_extension", "authenticated_bits", "leaky_ands", "bucketing", "triple_checks",
                     "circuit_input", "circuit_online", "circuit_output", "ideal_dealer", "manifest")
INFER_ONLINE_GROUPS = ("plane_evaluation", "online_local", "layer_openings", "final_check")
INFER_GROUPS = INFER_PREP_GROUPS + INFER_ONLINE_GROUPS

# --quick: a smoke run of every driver and of the summary on a few small configurations (one repetition each,
# output in experiments/quick); see quick_keep().  QUICK["on"] is set by main().
QUICK = {"on": False}
QUICK_GROUPS = "muklew"
QUICK_MODEL = ("mlp16", 16, 8)
QUICK_CORE_CIRCUITS = (f"setup_s{S}", f"gate_n1_k{K}_s{S}_shared_om1", f"dpf_n8_k{K}_s{S}_xor_one",
                       f"dpf_n16_k{K}_s{S}_xor_one", f"dcf_n8_k{K}_s{S}_xor_one")


# --------------------------------------------------------------------------- #
# Machine and machine load (the method of the core's campaign runner)
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _write_json(path: str, doc) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _stats(values: list) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _binary(name: str, build_dir: str) -> str:
    for cand in (name + ".exe", name):
        path = os.path.join(build_dir, cand)
        if os.path.isfile(path):
            return path
    raise SystemExit(f"{name} not found in {build_dir}: build it first (cpp/build_msvc.bat or cpp/build.sh)")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _powershell(command: str) -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                             capture_output=True, text=True, timeout=120)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_sets() -> dict:
    """Logical processors and cores per efficiency class (Windows; empty elsewhere)."""
    if os.name != "nt":
        return {}
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        need = ctypes.c_ulong(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
        buf = (ctypes.c_ubyte * need.value)()
        if not k32.GetSystemCpuSetInformation(buf, need.value, ctypes.byref(need), None, 0):
            return {}
        raw, pos, classes = bytes(buf), 0, {}
        while pos + 8 <= need.value:
            size = int.from_bytes(raw[pos:pos + 4], "little")
            if size < 20:
                break
            core, eff = raw[pos + 15], raw[pos + 18]
            entry = classes.setdefault(eff, {"logical_processors": 0, "cores": set()})
            entry["logical_processors"] += 1
            entry["cores"].add(core)
            pos += size
        return {str(eff): {"logical_processors": e["logical_processors"], "cores": len(e["cores"])}
                for eff, e in sorted(classes.items())}
    except (OSError, AttributeError, ValueError):
        return {}


def machine_info(compiler: str = "") -> dict:
    """Description of the machine from system queries (no user or host names)."""
    info = {"schema": "couplet.machine/1", "queried_utc": _utc_now()}
    if os.name == "nt":
        text = _powershell(
            "$p = Get-CimInstance Win32_Processor | Select-Object -First 1; "
            "$o = Get-CimInstance Win32_OperatingSystem; $c = Get-CimInstance Win32_ComputerSystem; "
            "$m = Get-CimInstance Win32_PhysicalMemory; "
            "@{cpu = $p.Name.Trim(); cores = $p.NumberOfCores; threads = $p.NumberOfLogicalProcessors; "
            "mhz = $p.MaxClockSpeed; sockets = @(Get-CimInstance Win32_Processor).Count; os = $o.Caption.Trim(); "
            "os_version = $o.Version; os_build = $o.BuildNumber; arch = $o.OSArchitecture; "
            "ram = $c.TotalPhysicalMemory; dimm = ($m | Measure-Object -Property Capacity -Sum).Sum; "
            "dimm_mhz = ($m | Select-Object -First 1).ConfiguredClockSpeed} | ConvertTo-Json -Compress")
        try:
            q = json.loads(text)
        except ValueError:
            q = {}
        info.update({
            "cpu": q.get("cpu"), "sockets": q.get("sockets"), "physical_cores": q.get("cores"),
            "hardware_threads": q.get("threads") or os.cpu_count(), "cpu_base_mhz": q.get("mhz"),
            "os": q.get("os"), "os_version": q.get("os_version"), "os_build": q.get("os_build"),
            "os_architecture": q.get("arch"),
            "ram_bytes_visible": int(q["ram"]) if q.get("ram") else None,
            "ram_bytes_installed": int(q["dimm"]) if q.get("dimm") else None,
            "ram_gib": round(int(q["dimm"]) / 2 ** 30) if q.get("dimm") else
            (round(int(q["ram"]) / 2 ** 30) if q.get("ram") else None),
            "ram_configured_mhz": q.get("dimm_mhz"),
        })
        sets = _cpu_sets()
        if sets:
            info["efficiency_classes"] = sets
            info["efficiency_classes_note"] = ("logical processors and cores per efficiency class as reported by "
                                               "GetSystemCpuSetInformation; a higher class is a faster core type")
    else:
        info.update({"cpu": platform.processor() or None, "hardware_threads": os.cpu_count(),
                     "os": platform.system(), "os_version": platform.release()})
        try:
            with open("/proc/cpuinfo", "r", encoding="ascii", errors="replace") as f:
                cores = set()
                phys = core = None
                for line in f:
                    key, _, val = line.partition(":")
                    key, val = key.strip(), val.strip()
                    if key == "model name" and val:
                        info["cpu"] = val
                    elif key == "physical id":
                        phys = val
                    elif key == "core id":
                        core = val
                        cores.add((phys, core))
                if cores:
                    info["physical_cores"] = len(cores)
                    info["sockets"] = len({p for p, _ in cores})
        except OSError:
            pass
        try:
            with open("/etc/os-release", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info["os"] = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        try:
            with open("/proc/meminfo", "r", encoding="ascii") as f:
                kb = int(f.readline().split()[1])
            info["ram_bytes_visible"] = kb * 1024
            info["ram_gib"] = round(kb / 2 ** 20)
        except (OSError, ValueError, IndexError):
            pass
    info["python"] = platform.python_version()
    info["python_implementation"] = platform.python_implementation() + " (" + platform.python_compiler() + ")"
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    if compiler:
        info["compiler"] = compiler
        if compiler.startswith("MSVC"):
            info["compiler_flags"] = "/O2 /std:c++17 /EHsc /arch:AVX2 /W4 /DNDEBUG"
        else:
            # written by scripts/build_linux.sh (cpp/build.sh) next to the programs
            try:
                with open(os.path.join(CPP, "build", "build_flags.txt"), "r", encoding="utf-8") as f:
                    info["compiler_flags"] = f.read().strip() or None
            except OSError:
                info["compiler_flags"] = None
    info["note"] = ("both parties of every run are processes of this one machine (127.0.0.1); the runs are not pinned; "
                    "the background load of the rest of the machine during each run is recorded in the rows")
    if os.name != "nt":
        info["note"] += ("; on this operating system the runner records no background load and no peak memory "
                         "(both are read from Windows counters), so every run counts as valid for timing")
    return info


def format_msvc(text: str) -> str:
    """'MSVC 195136257' (the _MSC_FULL_VER of the records) -> 'MSVC 19.51.36257'."""
    parts = text.split()
    if len(parts) == 2 and parts[0] == "MSVC" and parts[1].isdigit() and len(parts[1]) == 9:
        v = parts[1]
        return f"MSVC {v[:2]}.{v[2:4]}.{v[4:]}"
    return text


class _FileTimes(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


def _system_times():
    """(idle, total) processor time of the whole machine in 100 ns units, or None."""
    if os.name != "nt":
        return None
    idle, kernel, user = _FileTimes(), _FileTimes(), _FileTimes()
    if not ctypes.WinDLL("kernel32").GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    val = lambda t: (t.hi << 32) | t.lo                                      # noqa: E731
    return val(idle), val(kernel) + val(user)                                  # kernel time includes idle time


def background_cpu_percent(interval: float = 0.3):
    a = _system_times()
    if a is None:
        return None
    time.sleep(interval)
    b = _system_times()
    total = b[1] - a[1]
    return round(100.0 * (1.0 - (b[0] - a[0]) / total), 1) if total > 0 else None


QUIET = {"percent": 10.0, "percent_wan": 30.0, "max_wait": 900.0}


def threshold(cfg) -> float:
    """Largest background load of a run valid for timing: emulated-link rows (whose time is dominated by the
    emulated transfers and delays) use percent_wan, all others percent."""
    return QUIET["percent_wan"] if cfg and cfg.get("link", "lan") != "lan" else QUIET["percent"]


def wait_quiet(limit=None) -> dict:
    limit = QUIET["percent"] if limit is None else limit
    t0 = time.perf_counter()
    told = 0.0
    while True:
        load = background_cpu_percent(0.5)
        waited = time.perf_counter() - t0
        if load is None or load < limit or waited >= QUIET["max_wait"]:
            return {"background_cpu_percent_before": load, "quiet_wait_s": round(waited, 2)}
        if waited - told >= 600.0:
            told = waited
            print(f"    waiting for a quiet machine: {waited / 60:.0f} min, load {load}%", flush=True)


class _JobAccounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", ctypes.c_uint32), ("TotalProcesses", ctypes.c_uint32),
                ("ActiveProcesses", ctypes.c_uint32), ("TotalTerminatedProcesses", ctypes.c_uint32)]


class CpuWindow:
    """Processor time of the whole machine and of our process tree over a window (job object; Windows)."""

    def __init__(self):
        self.job = None
        self.t0 = _system_times()
        if os.name == "nt":
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.CreateJobObjectW.restype = ctypes.c_void_p
                k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
                self.job = k32.CreateJobObjectW(None, None) or None
            except (OSError, AttributeError):
                self.job = None
        self.assigned = self.job is not None

    def add(self, proc: subprocess.Popen) -> None:
        if self.job is None:
            return
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        if not k32.AssignProcessToJobObject(self.job, int(proc._handle)):
            self.assigned = False

    def close(self) -> dict:
        t1 = _system_times()
        out = {"background_cpu_percent_during": None, "machine_cpu_percent_during": None, "own_cpu_seconds": None}
        if self.job is None:
            return out
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.c_void_p]
        k32.QueryInformationJobObject.restype = ctypes.c_int
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        acc = _JobAccounting()
        ok = k32.QueryInformationJobObject(self.job, 1, ctypes.byref(acc), ctypes.sizeof(acc), None)
        k32.CloseHandle(self.job)
        self.job = None
        if not ok or not self.assigned or self.t0 is None or t1 is None:
            return out
        total, idle = t1[1] - self.t0[1], t1[0] - self.t0[0]
        own = acc.TotalUserTime + acc.TotalKernelTime
        if total <= 0:
            return out
        out["own_cpu_seconds"] = round(own / 1e7, 3)
        out["machine_cpu_percent_during"] = round(100.0 * (total - idle) / total, 1)
        out["background_cpu_percent_during"] = round(max(0.0, 100.0 * (total - idle - own) / total), 1)
        return out


def timing_valid(window: dict, cfg=None) -> bool:
    load = window.get("background_cpu_percent_during")
    return load is None or load <= threshold(cfg)


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_memory_of(proc: subprocess.Popen):
    """Peak working set and peak commit charge (bytes) of a finished child process, or None."""
    if os.name != "nt":
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemCounters), ctypes.c_uint32]
        k32.K32GetProcessMemoryInfo.restype = ctypes.c_int
        c = _MemCounters()
        c.cb = ctypes.sizeof(_MemCounters)
        if not k32.K32GetProcessMemoryInfo(int(proc._handle), ctypes.byref(c), c.cb):
            return None
        return {"peak_working_set_bytes": int(c.PeakWorkingSetSize), "peak_committed_bytes": int(c.PeakPagefileUsage)}
    except (OSError, AttributeError, ValueError):
        return None


def free_memory_bytes():
    if os.name != "nt":
        return None

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    st = Status()
    st.dwLength = ctypes.sizeof(Status)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return int(st.ullAvailPhys)


# --------------------------------------------------------------------------- #
# Models (group m)
# --------------------------------------------------------------------------- #

def model_name(preset: str, n: int, f: int) -> str:
    return f"{preset}_{n}_{f}"


def model_files(name: str) -> dict:
    return {"model": os.path.join(MODELS_DIR, name + "_model.json"),
            "vectors": os.path.join(MODELS_DIR, name + "_vectors.json"),
            "report": os.path.join(MODELS_DIR, name + "_report.json")}


def model_specs() -> list:
    """(preset, n, f, private plans wanted) of every model the campaign needs."""
    if QUICK["on"]:
        return [QUICK_MODEL + (False,)]
    specs = []
    for n, f in FORMATS:
        for preset in E2E_MODELS:
            specs.append((preset, n, f, preset in PRIVATE_MODELS and (n, f) == (16, 8)))
    for n, f in ONLINE_FORMATS:
        if (n, f) not in FORMATS:
            specs.append(("mlp784", n, f, False))
        specs.append(("mlp784h1", n, f, False))
        specs.append(("mlp784h3", n, f, False))
    return specs


def _evaluate_round(model, test, seed: int) -> dict:
    """Accuracy and float agreement of the mirror in mode round with uniform public thresholds."""
    import numpy as np
    from couplet import nn as nnm
    x, y = test
    rng = np.random.default_rng(int(seed))
    float_pred = np.argmax(model.forward_float(x), axis=1)
    pred = []
    for row in x:
        thr = [[int(v) for v in rng.integers(0, 1 << model.f, w)] for w in model.hidden_widths]
        pred.append(int(np.argmax(model.mirror(nnm.features_to_codes(row, model.f), "round", thr).logits)))
    pred = np.array(pred)
    return {"mirror_round_accuracy": float((pred == y).mean()), "prediction_agreement_round": float((pred == float_pred).mean()),
            "round_threshold_seed": int(seed)}


def train_and_export(preset: str, n: int, f: int) -> dict:
    """Train, export (model, 16 test inputs, table) and evaluate one model; returns its report."""
    import numpy as np
    from couplet.tools import export_model as em
    from couplet.tools import train_synthetic as ts
    name = model_name(preset, n, f)
    t0 = time.perf_counter()
    if preset in MODEL_SIZES:
        sizes = MODEL_SIZES[preset]
        tr, te = ts.make_dataset(sizes[0], sizes[-1], TRAIN["train"], TRAIN["test"], TRAIN["data_seed"])
        model = ts.train_mlp(sizes, "tanh", tr, TRAIN["epochs"], TRAIN["init_seed"], n, f)
        report = ts.evaluate(model, te)
        report.update({"preset": preset, "epochs": TRAIN["epochs"], "train": TRAIN["train"], "test": TRAIN["test"],
                       "data_seed": TRAIN["data_seed"], "init_seed": TRAIN["init_seed"]})
    else:
        model, te, report = ts.train_preset(preset, n, f, epochs=TRAIN["epochs"], train=TRAIN["train"],
                                            test=TRAIN["test"], data_seed=TRAIN["data_seed"],
                                            init_seed=TRAIN["init_seed"])
    report["train_and_exact_eval_seconds"] = round(time.perf_counter() - t0, 1)
    report.update(_evaluate_round(model, te, 1000 + n))
    rng = np.random.default_rng(TRAIN["input_seed"])
    idx = rng.permutation(len(te[0]))[:N_INPUTS]
    paths = em.export(model, MODELS_DIR, name, te[0][idx].tolist(), TRAIN["input_seed"])
    report.update({
        "schema": "couplet.model_report/1", "name": name, "kind": model.kind, "act": model.act, "n": n, "f": f,
        "sizes": [model.d] + [r for r, _ in model.shapes], "sites": int(sum(model.hidden_widths)),
        "hidden_widths": [int(w) for w in model.hidden_widths], "weights": int(sum(r * c for r, c in model.shapes)),
        "test_inputs_evaluated": int(len(te[0])), "exported_inputs": N_INPUTS,
        "exported_input_test_indices": [int(i) for i in idx], "files": [os.path.basename(p) for p in paths],
        "model_hash": model.model_hash().hex(), "public_dataset_used": False,
        "data": "seeded synthetic task of couplet.tools.train_synthetic (make_dataset): " +
                ("image task, d = image^2" if model.kind == "cnn" else "flat task"),
        "accuracy_note": "float_accuracy: the float model; mirror_exact_accuracy / mirror_round_accuracy: the fixed-point "
                         "mirror of the secure pipeline (couplet.nn.Model.mirror) in mode exact / round (uniform thresholds "
                         "from round_threshold_seed); prediction_agreement(_round): argmax of the mirror equal to argmax of "
                         "the float model; all on the whole synthetic test set",
        "generated_utc": _utc_now()})
    _write_json(model_files(name)["report"], report)
    return report


def export_plans(name: str, private: bool) -> str:
    """bench/export_circuits.py --plan for the model (public plans, and the private ones if asked); returns ''."""
    path = model_files(name)["model"]
    cmds = [[sys.executable, os.path.join(_HERE, "export_circuits.py"), "--plan", path, "--skip-existing"]]
    if private:
        cmds.append(cmds[0] + ["--private-model"])
    for cmd in cmds:
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=_PY)
        if out.returncode:
            return f"export_circuits failed for {name}: {(out.stderr or out.stdout)[-600:]}"
    return ""


def group_models(force: bool) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    rows, problems = [], []
    for preset, n, f, private in model_specs():
        name = model_name(preset, n, f)
        files = model_files(name)
        report = None if force else _read_json(files["report"])
        if report is None or not all(os.path.isfile(files[k]) for k in ("model", "vectors")):
            print(f"  training {name} ...", flush=True)
            report = train_and_export(preset, n, f)
            print(f"  {name}: float {report['float_accuracy']:.4f}, mirror exact {report['mirror_exact_accuracy']:.4f}, "
                  f"round {report['mirror_round_accuracy']:.4f}, agreement {report['prediction_agreement']:.4f} / "
                  f"{report['prediction_agreement_round']:.4f}, overflows {report['code_overflows']}", flush=True)
        err = export_plans(name, private)
        if err:
            problems.append(err)
        rows.append(report)
    return {"status": "failed" if problems else "ok", "reason": "; ".join(problems), "measured_utc": _utc_now(),
            "models_dir": "experiments/models", "rows": rows}


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #

def _link(link: str) -> dict:
    d, r = LINKS[link]
    return {"link": link, "delay_ms": d, "bandwidth_mbps": r}


def cfg_item(item: str, batch: int = 1, threads: int = 1, link: str = "lan") -> dict:
    return dict({"driver": "couplet_item", "item": item, "batch": batch, "prep": "real", "sigma": SIGMA,
                 "threads": threads}, **_link(link))


def cfg_party(task: str, n: int, auth: bool = False, batch: int = 1, threads: int = 1, link: str = "lan") -> dict:
    return dict({"driver": "couplet_party", "task": task, "n": n, "k": K, "s": S, "beta": "public", "auth": auth,
                 "batch": batch, "prep": "real", "sigma": SIGMA, "threads": threads}, **_link(link))


def cfg_infer(preset: str, n: int, f: int, mode: str, private: bool = False, threads: int = 1,
              link: str = "lan") -> dict:
    chunk = (CHUNK_AND_PRIVATE.get(preset) if private else None) or CHUNK_AND.get(preset, 0)
    return dict({"driver": "couplet_infer", "model": model_name(preset, n, f), "preset": preset, "n": n, "f": f,
                 "mode": mode, "private_model": private, "chunk_and": chunk, "prep": "real",
                 "sigma": SIGMA, "threads": threads}, **_link(link))


def _suffix(cfg: dict) -> str:
    out = f"_s{cfg['sigma']}"
    if cfg["link"] != "lan":
        d = cfg["delay_ms"]
        out += f"_d{int(d) if float(d).is_integer() else d}ms_bw{int(cfg['bandwidth_mbps'])}"
    if cfg["threads"] != 1:
        out += f"_t{cfg['threads']}"
    return out


def config_name(cfg: dict) -> str:
    if cfg["driver"] == "couplet_item":
        return f"item_{cfg['item']}_b{cfg['batch']}" + _suffix(cfg)
    if cfg["driver"] == "couplet_party":
        return (f"party_{cfg['task']}_n{cfg['n']}_k{cfg['k']}" + ("_auth" if cfg["auth"] else "") + f"_b{cfg['batch']}"
                + _suffix(cfg))
    name = f"infer_{cfg['model']}_{cfg['mode']}" + ("_private" if cfg["private_model"] else "")
    if cfg["chunk_and"]:
        name += f"_c{cfg['chunk_and']}"
    return name + _suffix(cfg)


def plan() -> list:
    """Every two-process configuration: {'name', 'groups', 'view', 'cfg'} in the order of execution."""
    items = []

    def add(group: str, cfg: dict, view: str) -> None:
        name = config_name(cfg)
        for it in items:
            if it["name"] == name:
                if group not in it["groups"]:
                    it["groups"].append(group)
                return
        items.append({"name": name, "groups": [group], "view": view, "cfg": cfg})

    # u: unit costs
    for n, f in UNIT_NF:
        for mode in MODES:
            for batch in (1, 64):
                add("u", cfg_item(f"site_n{n}_f{f}_k{K}_s{S}_{mode}", batch), "unit")
    add("u", cfg_item(f"triples1_k{K}_s{S}", 1), "unit")
    add("u", cfg_item(f"triples1_k{K}_s{S}", 64), "unit")
    add("u", cfg_item(f"mattriple_16x16_k{K}_s{S}", 1), "unit")
    add("u", cfg_item(f"inmask_p0_c1_k{K}_s{S}", 1), "unit")
    add("u", cfg_item(f"gate_n1_k{K}_s{S}_shared_om1", 1), "unit")
    for n, _ in UNIT_NF:
        add("u", cfg_party("dcf", n), "unit")
    for n in (8, 16):
        add("u", cfg_party("gate-cmp", n, auth=True), "unit")
    # k: key generation (this work; the baselines are a single-program group)
    for n in KEYGEN_N:
        add("k", cfg_party("dpf", n), "keygen")
    # e: end to end, no delay
    for n, f in FORMATS:
        for preset in E2E_MODELS:
            for mode in MODES:
                add("e", cfg_infer(preset, n, f, mode), "e2e")
                if preset == "mlp784":
                    add("o", cfg_infer(preset, n, f, mode), "e2e")
    for n, f in FORMATS:
        for preset in LARGEST:
            for mode in MODES:
                add("e", cfg_infer(preset, n, f, mode, threads=8), "e2e")
    # o: online layer
    for n, f in ONLINE_FORMATS:
        for mode in MODES:
            add("o", cfg_infer("mlp784", n, f, mode), "e2e")
            add("o", cfg_infer("mlp784h1", n, f, mode), "e2e")
            add("o", cfg_infer("mlp784h3", n, f, mode), "e2e")
    # p: private-model mode
    for preset in PRIVATE_MODELS:
        for mode in MODES:
            add("p", cfg_infer(preset, 16, 8, mode, private=True), "e2e")
    for preset in PRIVATE_FULL:
        for mode in MODES:
            add("p", cfg_infer(preset, 16, 8, mode, private=True), "e2e")
    # w: emulated wide-area links
    for link in ("wan20", "wan80"):
        for preset in E2E_MODELS:
            for mode in MODES:
                add("w", cfg_infer(preset, 16, 8, mode, link=link), "e2e")
    if QUICK["on"]:
        items = [it for it in items if quick_keep(it["cfg"])]
    return items


def quick_keep(cfg: dict) -> bool:
    """The configurations of --quick: site(8, 6) in both modes, one Beaver triple, one input mask, one output
    mask, a DCF key (n = 8), DPF keys (n = 8, 16), and the MLP [16,8,8,4] at (16, 8) in both modes without
    delay and in mode round on the 20 ms / 100 Mbit/s link; one thread, batch 1."""
    if cfg["threads"] != 1 or cfg.get("batch", 1) != 1:
        return False
    if cfg["driver"] == "couplet_item":
        return cfg["item"].startswith(("site_n8_f6_", "triples1_", "inmask_p0_c1_", "gate_n1_"))
    if cfg["driver"] == "couplet_party":
        return (cfg["task"], cfg["n"]) in (("dpf", 8), ("dpf", 16), ("dcf", 8))
    return cfg["model"] == model_name(*QUICK_MODEL) and not cfg["private_model"] and \
        (cfg["link"] == "lan" or (cfg["link"] == "wan20" and cfg["mode"] == "round"))


# --------------------------------------------------------------------------- #
# One two-process run
# --------------------------------------------------------------------------- #

def _command(binary: str, cfg: dict, role: int, port: int, out_file: str, rep: int) -> list:
    cmd = [binary, "--role", str(role), "--host", "127.0.0.1", "--port", str(port), "--sigma", str(cfg["sigma"]),
           "--delay-ms", repr(float(cfg["delay_ms"])), "--threads", str(cfg["threads"]), "--quiet", "--out", out_file,
           "--circuits", CIRCUITS, "--verify", "--timeout-ms", "600000"]
    if cfg["bandwidth_mbps"]:
        cmd += ["--bandwidth-mbps", repr(float(cfg["bandwidth_mbps"]))]
    if cfg["driver"] == "couplet_item":
        cmd += ["--item", cfg["item"], "--batch", str(cfg["batch"]), "--prep", cfg["prep"]]
    elif cfg["driver"] == "couplet_party":
        cmd += ["--task", cfg["task"], "--n", str(cfg["n"]), "--k", str(cfg["k"]), "--batch", str(cfg["batch"]),
                "--prep", cfg["prep"]]
        if cfg["task"] in ("dpf", "dcf"):
            cmd += ["--beta", cfg["beta"]]
        elif cfg["auth"]:
            cmd.append("--auth")
    else:
        files = model_files(cfg["model"])
        cmd += ["--model", files["model"], "--input", files["vectors"], "--input-index", str(rep % N_INPUTS),
                "--mode", cfg["mode"], "--prep", cfg["prep"]]
        if cfg["private_model"]:
            cmd.append("--private-model")
        if cfg["chunk_and"]:
            cmd += ["--chunk-and", str(cfg["chunk_and"])]
    return cmd


def _portable(cmd: list) -> list:
    """The command line without machine-specific paths (for the record)."""
    out = []
    for i, arg in enumerate(cmd):
        prev = cmd[i - 1] if i else ""
        if i == 0:
            out.append("build/" + os.path.splitext(os.path.basename(arg))[0])
        elif prev == "--circuits":
            out.append("artifact/circuits")
        elif prev in ("--model", "--input"):
            out.append("experiments/models/" + os.path.basename(arg))
        elif prev == "--out":
            out.append(os.path.basename(arg))
        else:
            out.append(arg)
    return out


def run_pair(binaries: dict, cfg: dict, out_dir: str, tag: str, rep: int, timeout: float) -> dict:
    """One run of both processes; returns {'ok', 'error', 'records', 'peak_memory', ...}."""
    binary = binaries[cfg["driver"]]
    port = free_port()
    files = [os.path.join(out_dir, f"{tag}_p{role}.tmp.json") for role in (0, 1)]
    for path in files:
        if os.path.exists(path):
            os.remove(path)
    quiet = wait_quiet(threshold(cfg))
    procs, cmds = [], []
    started = _utc_now()
    t0 = time.perf_counter()
    window = CpuWindow()
    for role in (0, 1):
        cmd = _command(binary, cfg, role, port, files[role], rep)
        cmds.append(_portable(cmd))
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        window.add(procs[-1])
        if role == 0:
            time.sleep(0.2)
    outs = []
    for p in procs:
        try:
            outs.append(p.communicate(timeout=timeout))
        except subprocess.TimeoutExpired:
            for other in procs:
                other.kill()
            for other in procs:
                other.communicate()
            window.close()
            return {"ok": False, "error": f"timeout after {timeout:.0f} s", "rep": rep}
    elapsed = time.perf_counter() - t0
    load = window.close()
    res = {"ok": False, "rep": rep, "started_utc": started, "processes_elapsed_s": round(elapsed, 3),
           "background_cpu_percent_before": quiet["background_cpu_percent_before"], "quiet_wait_s": quiet["quiet_wait_s"],
           "background_cpu_percent_during": load["background_cpu_percent_during"],
           "machine_cpu_percent_during": load["machine_cpu_percent_during"], "own_cpu_seconds": load["own_cpu_seconds"],
           "timing_valid": timing_valid(load, cfg), "commands": cmds, "peak_memory": [peak_memory_of(p) for p in procs]}
    codes = [p.returncode for p in procs]
    records = [_read_json(path) for path in files]
    for path in files:
        if os.path.exists(path):
            os.remove(path)
    if any(codes):
        res["error"] = f"exit codes {codes}: " + " | ".join((o[1] or o[0]).strip()[-300:] for o in outs)
        return res
    if any(r is None for r in records):
        res["error"] = "a party wrote no readable record"
        return res
    res["records"] = records
    problems = check_records(cfg, records)
    if not problems and cfg["driver"] == "couplet_infer":
        res["python_mirror"] = python_mirror_check(cfg, records, rep)
        if not res["python_mirror"]["equal"]:
            problems.append("the logits differ from the Python mirror")
    if problems:
        res["error"] = "; ".join(problems)
        return res
    res["ok"] = True
    return res


def _phase_names(cfg: dict) -> tuple:
    return INFER_GROUPS if cfg["driver"] == "couplet_infer" else PREP_PHASES + ENGINE_PHASES


def _phases(cfg: dict, rec: dict) -> dict:
    return rec["phase_groups"] if cfg["driver"] == "couplet_infer" else rec["phases"]


def check_records(cfg: dict, records: list) -> list:
    """The acceptance checks of one run (see the module docstring); returns the list of problems."""
    r0, r1 = records
    names = _phase_names(cfg)
    problems = []
    infer = cfg["driver"] == "couplet_infer"
    for r in records:
        ch, pre, par = r["checks"], r["preprocessing"], r["params"]
        if not ch["phase_sums_ok"]:
            problems.append("the party reports that its counters do not sum to the totals")
        ph = _phases(cfg, r)
        if set(ph) != set(names):
            problems.append("unexpected set of phases " + ",".join(sorted(set(ph) ^ set(names))))
            continue
        ver = r["verification"]
        if not ver["performed"] or (ver.get("logits_correct") if infer else ver.get("outputs_correct")) is not True:
            problems.append("verification")
        if cfg["driver"] == "couplet_item" and ver.get("negative_control_rejected") is not True:
            problems.append("the negative control of the material check was not rejected")
        if not ch["all_triples_used"] or not ch["strict_mode"] or pre.get("allow_refill") is not False:
            problems.append("the triples were not used up in strict mode")
        chunked = infer and cfg["chunk_and"]
        if not chunked:
            if not ch["single_bucketing_population"] or pre.get("triple_requests") != 1:
                problems.append("bucketing population (triple_requests != 1)")
            if pre.get("triples_requested") != pre["triples"]:
                problems.append("the one triple request is not exactly the triples of the session")
        req = pre.get("triple_requests") or 0
        if req < 1 or abs(pre.get("bucketing_bound_log2") - (math.log2(req) - cfg["sigma"])) > 1e-6:
            problems.append("bucketing bound is not log2(requests) - sigma")
        if r.get("ideal_preprocessing"):
            problems.append("ideal preprocessing")
        if par["sigma"] != cfg["sigma"] or par["threads"] != cfg["threads"] or \
                abs(float(par["delay_ms"]) - cfg["delay_ms"]) > 1e-9 or \
                abs(float(par.get("bandwidth_mbps") or 0.0) - cfg["bandwidth_mbps"]) > 1e-9:
            problems.append("the record does not describe the requested configuration")
        if infer and (par["mode"] != cfg["mode"] or par["private_model"] != cfg["private_model"] or
                      par["chunk_and"] != cfg["chunk_and"]):
            problems.append("the record does not describe the requested model run")
        for field in ("bytes_sent", "bytes_recv", "rounds"):
            if sum(ph[p][field] for p in names) != r["totals"][field]:
                problems.append(f"phase {field} do not sum to the total")
        if infer and sum(ph[p]["bytes_sent"] for p in INFER_ONLINE_GROUPS) != r["online"]["bytes_sent"]:
            problems.append("the online groups do not give the online bytes")
    if problems:
        return sorted(set(problems))
    p0, p1 = _phases(cfg, r0), _phases(cfg, r1)
    if r0["totals"]["bytes_sent"] != r1["totals"]["bytes_recv"] or r0["totals"]["bytes_recv"] != r1["totals"]["bytes_sent"]:
        problems.append("sent != received")
    for p in names:
        if p0[p]["bytes_sent"] != p1[p]["bytes_recv"] or p0[p]["bytes_recv"] != p1[p]["bytes_sent"]:
            problems.append(f"phase {p}: sent != received")
        if p0[p]["rounds"] != p1[p]["rounds"]:
            problems.append(f"phase {p}: round counts differ")
    if r0["totals"]["rounds"] != r1["totals"]["rounds"]:
        problems.append("round counts differ")
    if r0["preprocessing"]["triples"] != r1["preprocessing"]["triples"] or \
            r0["preprocessing"]["triple_requests"] != r1["preprocessing"]["triple_requests"]:
        problems.append("the parties disagree on the triples")
    if infer:
        if r0["manifest"]["digest"] != r1["manifest"]["digest"]:
            problems.append("manifest digests differ")
        if r0["logits"] != r1["logits"]:
            problems.append("the parties output different logits")
    elif r0["circuit"]["digest"] != r1["circuit"]["digest"]:
        problems.append("circuit digests differ")
    return sorted(set(problems))


_MODEL_CACHE: dict = {}


def _model_and_vectors(name: str):
    if name not in _MODEL_CACHE:
        from couplet import nn as nnm
        files = model_files(name)
        _MODEL_CACHE[name] = (nnm.Model.load(files["model"]), _read_json(files["vectors"]))
    return _MODEL_CACHE[name]


def python_mirror_check(cfg: dict, records: list, rep: int) -> dict:
    """The logits of a run against the Python mirror (couplet.nn) on the run's input and thresholds."""
    model, vec = _model_and_vectors(cfg["model"])
    idx = rep % N_INPUTS
    codes = vec["inputs"][idx]
    logits = [int(v) for v in records[0]["logits"]]
    thr = records[1]["online"]["thresholds"] if cfg["mode"] == "round" else None
    mirror = [int(v) for v in model.mirror(codes, cfg["mode"], thr).logits]
    exact = [int(v) for v in vec["exact_logits"][idx]]
    dev = max(abs(a - b) for a, b in zip(logits, exact))
    return {"input_index": idx, "equal": logits == mirror, "max_abs_deviation_from_exact_mirror": dev,
            "argmax_equals_exact_mirror": int(max(range(len(logits)), key=lambda i: logits[i])) ==
            int(max(range(len(exact)), key=lambda i: exact[i])),
            "float_argmax": None if vec.get("float_logits") is None else
            int(max(range(len(vec["float_logits"][idx])), key=lambda i: vec["float_logits"][idx][i])),
            "secure_argmax": int(max(range(len(logits)), key=lambda i: logits[i]))}


def fingerprint(cfg: dict, records: list) -> tuple:
    """Everything that must be identical in all runs: bytes and rounds, total and per phase."""
    r0, r1 = records
    p0, p1 = _phases(cfg, r0), _phases(cfg, r1)
    return (r0["totals"]["bytes_sent"], r1["totals"]["bytes_sent"], r0["totals"]["rounds"], r0["preprocessing"]["triples"],
            r0["preprocessing"]["triple_requests"],
            tuple((p0[p]["bytes_sent"], p1[p]["bytes_sent"], p0[p]["rounds"]) for p in _phase_names(cfg)))


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #

def repetitions_for(seconds: float, repeat: int) -> int:
    """`repeat` below 10 s, 5 up to a minute, 3 above."""
    return repeat if seconds < 10.0 else (min(repeat, 5) if seconds <= 60.0 else min(repeat, 3))


def _seconds(run: dict) -> float:
    return max(r["totals"]["seconds"] for r in run["records"])


def _valid(run: dict, cfg=None) -> bool:
    """Valid for timing under the threshold of THIS invocation (recomputed, so all rows of a kind use one criterion)."""
    load = run.get("background_cpu_percent_during")
    return load is None or load <= threshold(cfg)


def timed_runs(doc: dict) -> list:
    valid = [run for run in doc["runs"] if _valid(run, doc.get("config"))]
    return valid if len(valid) >= min(3, len(doc["runs"])) else list(doc["runs"])


def run_config(binaries: dict, item: dict, out_dir: str, repeat: int, timeout: float) -> dict:
    cfg, name = item["cfg"], item["name"]
    doc = {"schema": "couplet.m3.campaign.config/1", "name": name, "groups": item["groups"], "view": item["view"],
           "config": cfg, "driver": cfg["driver"], "started_utc": _utc_now(), "runs": [], "status": "ok", "reason": ""}
    path = os.path.join(out_dir, name + ".json")

    def fail(reason: str) -> dict:
        doc["status"], doc["reason"] = "failed", reason
        _write_json(path, doc)
        print(f"FAILED {name}: {reason}", flush=True)
        return doc

    reps, first, valid, rep, attempts_left = repeat, None, 0, 0, None
    while valid < reps:
        res = run_pair(binaries, cfg, out_dir, f"{name}_run{rep}", rep, timeout)
        doc["runs"].append(res)
        if not res["ok"]:
            return fail(f"run {rep}: {res['error']}")
        fp = fingerprint(cfg, res["records"])
        if first is None:
            first = fp
        elif fp != first:
            return fail(f"run {rep}: bytes or rounds differ from run 0")
        t = _seconds(res)
        if attempts_left is None:
            reps = repetitions_for(t, repeat)
            attempts_left = 3 * reps
        attempts_left -= 1
        valid += 1 if res["timing_valid"] else 0
        r0 = res["records"][0]
        print(f"  {name} run {rep}: {r0['totals']['bytes']:,} bytes, {r0['totals']['rounds']} rounds, {t:.3f} s, "
              f"verified, background {res['background_cpu_percent_during']}%"
              + ("" if res["timing_valid"] else " (not valid for timing)"), flush=True)
        rep += 1
        if attempts_left <= 0 and valid < reps:
            doc["timing_disturbed"] = True
            print(f"  {name}: only {valid} of {reps} runs were valid for timing", flush=True)
            break
    doc["finished_utc"] = _utc_now()
    timed = timed_runs(doc)
    secs = [_seconds(run) for run in timed]
    order = sorted(range(len(secs)), key=lambda i: secs[i])
    doc["median_run"] = timed[order[len(order) // 2]]["rep"]
    doc["repetitions"], doc["attempts"] = len(timed), len(doc["runs"])
    _write_json(path, doc)
    for role in (0, 1):
        _write_json(os.path.join(out_dir, f"{name}_p{role}.json"), doc["runs"][doc["median_run"]]["records"][role])
    return doc


# --------------------------------------------------------------------------- #
# Rows of the summary
# --------------------------------------------------------------------------- #

def _slower(run: dict, getter) -> float:
    return max(getter(run["records"][0]), getter(run["records"][1]))


def _med(runs: list, getter) -> dict:
    return _stats([_slower(run, getter) for run in runs])


def build_row(doc: dict) -> dict:
    """One row of the summary from the file of a configuration."""
    cfg, name = doc["config"], doc["name"]
    row = {"id": f"{doc['view']}/{name}", "name": name, "groups": sorted(doc["groups"]), "view": doc["view"],
           "status": doc["status"], "driver": doc["driver"],
           "config": {k: v for k, v in cfg.items() if k != "driver"}}
    if doc["status"] != "ok":
        row["reason"] = doc.get("reason", "")
        for key in ("estimate", "estimate_note"):
            if key in doc:
                row[key] = doc[key]
        return row
    runs = timed_runs(doc)
    secs = [_seconds(run) for run in runs]
    mid = sorted(range(len(secs)), key=lambda i: secs[i])[len(secs) // 2]
    r0, r1 = runs[mid]["records"]
    names = _phase_names(cfg)
    p0, p1 = _phases(cfg, r0), _phases(cfg, r1)
    total = r0["totals"]["bytes_sent"] + r1["totals"]["bytes_sent"]
    phases = {}
    for p in names:
        st = _med(runs, lambda r, p=p: _phases(cfg, r)[p]["seconds"])
        phases[p] = {"bytes": p0[p]["bytes_sent"] + p1[p]["bytes_sent"], "bytes_sent_p0": p0[p]["bytes_sent"],
                     "bytes_sent_p1": p1[p]["bytes_sent"], "rounds": p0[p]["rounds"], "seconds_median": st["median"],
                     "seconds_min": st["min"], "seconds_max": st["max"]}
    pre = r0["preprocessing"]
    mems = [m for run in runs for m in run.get("peak_memory") or [] if m]
    row.update({
        "and_gates": pre["triples"], "triples_session_setup": pre["triples_session_setup"],
        "bucket_size": pre["bucket_size"], "triple_requests": pre["triple_requests"],
        "bucketing_bound_log2": pre["bucketing_bound_log2"],
        "bytes": total, "bytes_sent_p0": r0["totals"]["bytes_sent"], "bytes_sent_p1": r1["totals"]["bytes_sent"],
        "bytes_per_party_max": max(r0["totals"]["bytes_sent"], r1["totals"]["bytes_sent"]),
        "bytes_per_and_gate": total / pre["triples"], "rounds": r0["totals"]["rounds"],
        "seconds": secs[mid], "seconds_min": min(secs), "seconds_max": max(secs), "seconds_all": secs,
        "phases": phases,
        "phase_bytes_sum_ok": sum(ph["bytes"] for ph in phases.values()) == total,
        "phase_rounds_sum_ok": sum(ph["rounds"] for ph in phases.values()) == r0["totals"]["rounds"],
        "bytes_identical_across_runs": True,
        "emulated_wait_s": max(r0["totals"].get("emulated_wait_s", 0.0), r1["totals"].get("emulated_wait_s", 0.0)),
        "bandwidth_wait_s": max(r0["totals"].get("bandwidth_wait_s", 0.0), r1["totals"].get("bandwidth_wait_s", 0.0)),
        "peak_memory": None if not mems else {
            "max_working_set_bytes": max(m["peak_working_set_bytes"] for m in mems),
            "max_committed_bytes": max(m["peak_committed_bytes"] for m in mems),
            "working_set_bytes_per_and_gate": max(m["peak_working_set_bytes"] for m in mems) / pre["triples"],
            "source": "operating-system counters of both processes, maximum over the timed runs (includes the loaded "
                      "circuit files and the small TEST-ONLY verification after the measured part)"},
        "verified": True, "verified_runs": len(doc["runs"]), "repetitions": len(runs), "attempts": len(doc["runs"]),
        "runs_excluded_from_timing": len(doc["runs"]) - len(runs),
        "timing_disturbed": bool(doc.get("timing_disturbed")) or any(not _valid(run, cfg) for run in runs),
        "median_run": runs[mid]["rep"], "timing_threshold_percent": threshold(cfg),
        "background_cpu_percent_during_runs": [run.get("background_cpu_percent_during") for run in runs],
        "median_over_min": round(statistics.median(secs) / min(secs), 3), "measured_utc": doc.get("finished_utc"),
    })
    if cfg["driver"] != "couplet_infer":
        circ = r0["circuit"]
        row.update({
            "circuit": circ["name"], "digest": circ["digest"], "n_and": circ["stats"]["n_and"],
            "and_depth": circ["stats"]["and_depth"], "n_gates": circ["stats"]["n_gates"], "batch": cfg["batch"],
            "bytes_per_instance": total / cfg["batch"], "seconds_per_instance": secs[mid] / cfg["batch"],
            "setup_s": _med(runs, lambda r: r["totals"]["setup_s"])["median"],
            "generation_s": _med(runs, lambda r: r["totals"]["generation_s"])["median"],
            "output_kind": r0["outputs"]["kind"],
            "output_bytes_per_instance_and_party": r0["outputs"]["bytes_per_instance_and_party"],
            "points_checked": r0["verification"]["points_checked"],
        })
        return row
    # couplet_infer: preprocessing and online parts, correctness
    on0 = sum(p0[p]["bytes_sent"] for p in INFER_ONLINE_GROUPS)
    on1 = sum(p1[p]["bytes_sent"] for p in INFER_ONLINE_GROUPS)
    on_rounds = sum(p0[p]["rounds"] for p in INFER_ONLINE_GROUPS)
    prep_s = _med(runs, lambda r: r["totals"]["setup_s"] + r["totals"]["preprocess_s"])
    online_s = _med(runs, lambda r: r["totals"]["infer_s"])
    mirror = [run.get("python_mirror") or {} for run in doc["runs"]]
    model, vec = _model_and_vectors(cfg["model"])
    row.update({
        "model": cfg["model"], "preset": cfg["preset"], "kind": r0["params"]["kind"], "act": r0["params"]["act"],
        "sizes": r0["params"]["sizes"], "sites": r0["model"]["sites"], "weights": r0["model"]["weights"],
        "manifest_digest": r0["manifest"]["digest"], "executions": r0["preprocessing"]["executions"],
        "chunk_and": cfg["chunk_and"],
        "prep_bytes": total - on0 - on1, "prep_bytes_p0": r0["totals"]["bytes_sent"] - on0,
        "prep_bytes_p1": r1["totals"]["bytes_sent"] - on1,
        "prep_bytes_per_party_max": max(r0["totals"]["bytes_sent"] - on0, r1["totals"]["bytes_sent"] - on1),
        "prep_rounds": r0["totals"]["rounds"] - on_rounds,
        "prep_seconds": prep_s["median"], "prep_seconds_min": prep_s["min"], "prep_seconds_max": prep_s["max"],
        "online_bytes": on0 + on1, "online_bytes_p0": on0, "online_bytes_p1": on1, "online_rounds": on_rounds,
        "online_seconds": online_s["median"], "online_seconds_min": online_s["min"], "online_seconds_max": online_s["max"],
        "plane_eval_seconds": _med(runs, lambda r: r["online"]["plane_eval_s"])["median"],
        "table_read_seconds": _med(runs, lambda r: r["online"]["table_read_s"])["median"],
        "online_local_seconds": _med(runs, lambda r: r["phase_groups"]["online_local"]["seconds"])["median"],
        "openings": r0["online"]["openings"], "checks": r0["online"]["checks"],
        "release_masked": r0["online"]["release_masked"],
        "bytes_per_site": total / max(1, r0["model"]["sites"]),
        "inputs_run": sorted({m.get("input_index") for m in mirror if m}),
        "bit_exact_runs": sum(1 for run in doc["runs"] if run["records"][0]["verification"]["logits_correct"] and
                              (run.get("python_mirror") or {}).get("equal")),
        "runs_checked": len(doc["runs"]),
        "max_abs_deviation_from_exact_mirror_ulp": max(m.get("max_abs_deviation_from_exact_mirror", 0) for m in mirror),
        "argmax_equals_exact_mirror_runs": sum(1 for m in mirror if m.get("argmax_equals_exact_mirror")),
        "secure_argmax_equals_float_runs": sum(1 for m in mirror if m.get("float_argmax") is not None and
                                               m.get("secure_argmax") == m.get("float_argmax")),
        "correctness_note": "bit_exact_runs: runs whose logits equal the driver's local check AND the Python mirror "
                            "(mode exact: the exact mirror; mode round: the mirror on the thresholds of the run); the "
                            "deviation is that of the logits from the exact mirror in units of 2^-2f (0 in mode exact)",
        "two_f": 2 * model.f,
    })
    return row


def _layer_diff(big: dict, small: dict) -> dict:
    """Online figures of `big` minus those of `small` (one more hidden layer of 128 sites)."""
    return {"layer_bytes": big["online_bytes"] - small["online_bytes"],
            "layer_bytes_per_party": max(big["online_bytes_p0"] - small["online_bytes_p0"],
                                         big["online_bytes_p1"] - small["online_bytes_p1"]),
            "layer_rounds": big["online_rounds"] - small["online_rounds"],
            "layer_seconds": big["online_seconds"] - small["online_seconds"],
            "layer_plane_eval_seconds": big["plane_eval_seconds"] - small["plane_eval_seconds"],
            "layer_table_read_seconds": big["table_read_seconds"] - small["table_read_seconds"]}


def derive_online_layers(e2e_rows: list) -> list:
    """Per hidden layer and per site from MLPs that differ by one hidden layer of 128 sites.

    The main figures (layer_*) are [784,128,128,128,10] minus [784,128,128,10]: both have authenticated
    hidden layers with deferred openings, so the difference is one such layer (its openings, plane
    evaluations, table reads and its 128 x 128 linear map).  first_check_* is [784,128,128,10] minus
    [784,128,10], which also contains the first ArithCheck batch of the release (the model with one hidden
    layer opens only by commit-then-open and has no deferred opening to check).
    """
    by = {(r["config"]["model"], r["config"]["mode"]): r for r in e2e_rows
          if r["status"] == "ok" and r["config"]["threads"] == 1 and r["config"]["link"] == "lan" and
          not r["config"]["private_model"]}
    out = []
    for n, f in ONLINE_FORMATS:
        for mode in MODES:
            three = by.get((model_name("mlp784h3", n, f), mode))
            full = by.get((model_name("mlp784", n, f), mode))
            one = by.get((model_name("mlp784h1", n, f), mode))
            row = {"n": n, "f": f, "mode": mode, "sites_per_layer": 128,
                   "three_hidden_layers": None if three is None else three["id"],
                   "full": None if full is None else full["id"], "one_hidden_layer": None if one is None else one["id"]}
            if full is None or three is None:
                row["status"] = "missing"
                out.append(row)
                continue
            row["status"] = "ok"
            row.update(_layer_diff(three, full))
            if one is not None:
                row.update({"first_check_" + k: v for k, v in _layer_diff(full, one).items()})
            row.update({
                "full_online_bytes": full["online_bytes"], "full_online_rounds": full["online_rounds"],
                "full_online_seconds": full["online_seconds"], "full_sites": full["sites"],
                "full_plane_eval_seconds": full["plane_eval_seconds"], "full_table_read_seconds": full["table_read_seconds"],
                "full_online_local_seconds": full["online_local_seconds"],
                "plane_eval_seconds_per_site": full["plane_eval_seconds"] / full["sites"],
                "table_read_seconds_per_site": full["table_read_seconds"] / full["sites"],
            })
            row["layer_bytes_per_site"] = row["layer_bytes"] / 128
            row["layer_seconds_per_site"] = row["layer_seconds"] / 128
            row["note"] = ("layer_* = MLP [784,128,128,128,10] minus MLP [784,128,128,10] (same task and seeds): one "
                           "authenticated hidden layer of 128 sites with its 128 x 128 linear map; seconds are "
                           "differences of medians; per-site plane times are those of the full MLP divided by its 256 "
                           "sites; first_check_* = [784,128,128,10] minus [784,128,10], which also holds the first "
                           "ArithCheck batch of the release")
            out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Infeasibility estimates
# --------------------------------------------------------------------------- #

def _lan_row_doc(out_dir: str, cfg: dict):
    ref = dict(cfg, link="lan", delay_ms=0.0, bandwidth_mbps=0.0)
    doc = _read_json(os.path.join(out_dir, config_name(ref) + ".json"))
    return doc if doc and doc.get("status") == "ok" else None


def estimate_wan(out_dir: str, item: dict, max_run_s: float, repeat: int):
    """An 'infeasible' document for a wide-area configuration that would exceed the limit, else None."""
    cfg = item["cfg"]
    ref = _lan_row_doc(out_dir, cfg)
    if ref is None:
        return {"schema": "couplet.m3.campaign.config/1", "name": item["name"], "groups": item["groups"],
                "view": item["view"], "config": cfg, "driver": cfg["driver"], "runs": [], "status": "not_run",
                "reason": "no accepted no-delay row of the same model, format and mode to estimate the run time from "
                          "(run group e first)"}
    run = ref["runs"][ref["median_run"]]
    r0, r1 = run["records"]
    rate = cfg["bandwidth_mbps"] * 1e6 / 8.0
    transfer = max(r0["totals"]["bytes_sent"] + 4 * r0["totals"]["msgs_sent"],
                   r1["totals"]["bytes_sent"] + 4 * r1["totals"]["msgs_sent"]) / rate
    lan_s = _seconds(run)
    est = lan_s + transfer + r0["totals"]["rounds"] * cfg["delay_ms"] / 1000.0
    reps = repetitions_for(est, repeat)
    estimate = {"from": ref["name"], "lan_seconds": lan_s, "transfer_seconds_per_direction_at_rate": transfer,
                "rounds": r0["totals"]["rounds"], "estimated_seconds_per_run": est, "repetitions": reps,
                "estimated_seconds_all_repetitions": est * reps, "limit_seconds_per_run": max_run_s}
    if est <= max_run_s:
        return None
    return {"schema": "couplet.m3.campaign.config/1", "name": item["name"], "groups": item["groups"], "view": item["view"],
            "config": cfg, "driver": cfg["driver"], "runs": [], "status": "infeasible", "estimate": estimate,
            "estimate_note": "estimate = seconds of the no-delay row + (payload + 4 framing bytes per message of the "
                             "larger direction) / rate + rounds x one-way delay; not a measurement",
            "reason": f"estimated {est:,.0f} s per run ({reps} repetitions: {est * reps / 3600:.1f} h) exceeds the limit "
                      f"of {max_run_s:,.0f} s per run of the campaign; the bandwidth alone needs {transfer:,.0f} s per run"}


def plan_file(cfg: dict) -> str:
    """The plan file that bench/export_circuits.py --plan writes for an inference configuration (group m)."""
    return os.path.join(MODELS_DIR, f"{cfg['model']}_plan_{cfg['mode']}" + ("_private" if cfg["private_model"] else "")
                        + f"_k{K}_s{S}.json")


def private_full_assessment(out_dir: str) -> list:
    """Not-run documents of the PRIVATE_FULL configurations that have no file in out_dir (task R-PRIV).

    A configuration with a file (accepted, failed or not run) is left to its file.  The AND gates are those of the
    configuration's plan file (the tiled plan of task R-TILE; the driver's manifest is the same plan, checked by
    test_ppml); seconds, bytes and memory are scaled from the accepted private-model row of the same model in the
    other mode if there is one (same cap, same circuits apart from the sites), else from the public-model row of the
    same model and mode.  An estimate, not a measurement."""
    docs = []
    for preset in PRIVATE_FULL:
        for mode in MODES:
            cfg = cfg_infer(preset, 16, 8, mode, private=True)
            name = config_name(cfg)
            if os.path.exists(os.path.join(out_dir, name + ".json")):
                continue
            doc = {"schema": "couplet.m3.campaign.config/1", "name": name, "groups": ["p"], "view": "e2e", "config": cfg,
                   "driver": "couplet_infer", "runs": [], "status": "not_run"}
            pf = plan_file(cfg)
            plan_doc = _read_json(pf)
            if plan_doc is None:
                doc["reason"] = f"not run; no plan file {os.path.basename(pf)} to estimate from (run group m first)"
                docs.append(doc)
                continue
            by_kind: dict = {}
            for e in plan_doc["entries"]:
                kind = e["descriptor"]["kind"]
                by_kind[kind] = by_kind.get(kind, 0) + e["n_and"] * e["batch"]
            need = int(plan_doc["and_budget"])
            tiling = plan_doc.get("tiling") or {}
            est = {"and_gates_needed": need, "and_gates_by_kind": by_kind, "plan_file": "experiments/models/" +
                   os.path.basename(pf), "plan_digest": plan_doc["digest"], "tiled": tiling.get("tiled", False),
                   "tiles": tiling.get("tiles"), "chunk_and": cfg["chunk_and"]}
            other = "exact" if mode == "round" else "round"
            basis = None
            for ref_cfg in (cfg_infer(preset, 16, 8, other, private=True), cfg_infer(preset, 16, 8, mode)):
                ref = _read_json(os.path.join(out_dir, config_name(ref_cfg) + ".json"))
                if ref and ref.get("status") == "ok":
                    basis = build_row(ref)
                    break
            if basis is None:
                doc["estimate"] = est
                doc["reason"] = (f"not run; the plan needs {need:,} AND gates; no accepted row of {preset} at (16, 8) to "
                                 f"scale the time from")
                docs.append(doc)
                continue
            rate = basis["and_gates"] / basis["prep_seconds"]
            prep_s = need / rate
            est.update({"basis": basis["id"], "basis_seconds": basis["seconds"], "basis_threads": basis["config"]["threads"],
                        "and_gates_per_second_of_basis": rate, "estimated_prep_seconds": prep_s,
                        "estimated_seconds_per_run": prep_s + basis["online_seconds"],
                        "estimated_prep_bytes_per_party": need * basis["prep_bytes_per_party_max"] / basis["and_gates"],
                        "estimated_bytes_both_directions": need * basis["bytes_per_and_gate"]})
            if basis["config"]["private_model"] and basis.get("peak_memory"):
                est["expected_peak_working_set_bytes"] = basis["peak_memory"]["max_working_set_bytes"]
            doc["estimate"] = est
            why = (f"the run of mode {other} ({basis['name']}) took {basis['seconds']:,.0f} s (median), and task R-PRIV "
                   f"(design/CHANGELOG_M3.md) ran mode {mode} only if one run of mode {other} took under 3,600 s"
                   if basis["config"]["private_model"] else "no run of this configuration yet")
            doc["reason"] = (f"not run: {why}; estimated {est['estimated_seconds_per_run']:,.0f} s per run at "
                             f"{basis['config']['threads']} thread(s) per party ({need:,} AND gates of the plan at the "
                             f"{rate / 1e6:.3f} x 10^6 AND gates per second of {basis['name']}) and "
                             f"{est['estimated_prep_bytes_per_party'] / 1e9:,.0f} GB of preprocessing per party")
            doc["estimate_note"] = ("AND gates from the plan file (exact); seconds, bytes and memory scaled from the basis "
                                    "row; not a measurement")
            docs.append(doc)
    return docs


def machine_ram_bytes():
    info = _read_json(os.path.join(RAW, "machine.json")) or {}
    return info.get("ram_bytes_visible") or info.get("ram_bytes_installed") or 0


# --------------------------------------------------------------------------- #
# Single-program groups
# --------------------------------------------------------------------------- #

def _run_tool(cmd: list, timeout: float, cwd=None, attempts_max: int = 6) -> tuple:
    """Run a benchmark program again while the background load during the run exceeds the limit."""
    attempts = []
    while True:
        quiet = wait_quiet()
        t0 = time.perf_counter()
        window = CpuWindow()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
        window.add(proc)
        try:
            out = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            window.close()
            return 1, f"timeout after {timeout:.0f} s", {"attempts": attempts}
        secs = time.perf_counter() - t0
        load = window.close()
        attempts.append({"background_cpu_percent_before": quiet["background_cpu_percent_before"],
                         "background_cpu_percent_during": load["background_cpu_percent_during"],
                         "seconds": round(secs, 1), "timing_valid": timing_valid(load)})
        print(f"    {os.path.basename(cmd[0])}: {secs:.1f} s, background {load['background_cpu_percent_during']}%", flush=True)
        if proc.returncode or attempts[-1]["timing_valid"] or len(attempts) >= attempts_max:
            return proc.returncode, (out[1] or out[0]).strip()[-600:], {
                "attempts": attempts, "timing_disturbed": not attempts[-1]["timing_valid"]}


def group_local(build_dir: str) -> dict:
    path = os.path.join(RAW, "bench_ppml_local.json")
    quick = ["--quick", "--ns", "8,16", "--threads", "1"] if QUICK["on"] else []
    code, tail, load = _run_tool([_binary("bench_ppml", build_dir), "--out", path] + quick, 3600)
    doc = _read_json(path)
    if code or not doc:
        return {"status": "failed", "reason": f"bench_ppml exit {code}: {tail}"}
    if not all(r.get("check_ok") for r in doc["rows"]):
        return {"status": "failed", "reason": "bench_ppml did not verify its reads"}
    return {"status": "ok", "file": "raw/bench_ppml_local.json", "machine_load": load, "measured_utc": _utc_now(),
            "rows": doc["rows"]}


def group_baselines(build_dir: str) -> dict:
    path = os.path.join(RAW, "baselines_cpp.json")
    ns = (8, 16) if QUICK["on"] else KEYGEN_N
    code, tail, load = _run_tool([_binary("bench_baselines", build_dir), path, "--n", ",".join(str(n) for n in ns)]
                                 + (["--reps", "2"] if QUICK["on"] else []), 7200)
    doc = _read_json(path)
    if code or not doc:
        return {"status": "failed", "reason": f"bench_baselines exit {code}: {tail}"}
    if not doc.get("all_outputs_verified") or not doc.get("all_counts_match_formulas"):
        return {"status": "failed", "reason": "bench_baselines did not verify its outputs or its counts"}
    return dict(baseline_summary(doc), machine_load=load, measured_utc=_utc_now())


CORRELATION_PHASES = ("cotext", "abit", "sh-cot")   # OT extension / authenticated bits / the unchecked COT rows


def baseline_summary(doc: dict) -> dict:
    """The rows of group k from raw/baselines_cpp.json (no measurement here; also used by --summary-only)."""
    rows = []
    for proto in ("leaky_zhang", "semihonest_ds"):
        for entry in doc.get(proto, []):
            for transport in ("tcp_two_processes", "loopback_two_threads"):
                rec = entry.get(transport)
                if not rec:
                    continue
                key, base = rec["per_key_without_base_ot"], rec["base_ot"]
                session = [max(run["key_seconds_p0"] + run["base_ot_seconds_p0"],
                               run["key_seconds_p1"] + run["base_ot_seconds_p1"]) for run in rec["runs"]]
                ph = rec.get("phases", {})
                party = [sum(v.get(f"bytes_p{b}", 0) for v in ph.values()) for b in (0, 1)]
                msgs = [sum(v.get(f"bytes_p{b}", 0) for name, v in ph.items()
                            if name != "baseot" and name not in CORRELATION_PHASES) for b in (0, 1)]
                abits = [sum(v.get(f"bytes_p{b}", 0) for name, v in ph.items() if name in CORRELATION_PHASES)
                         for b in (0, 1)]
                rows.append({
                    "id": f"baseline/{proto}_n{entry['n']}_{transport}", "status": "ok", "protocol": proto,
                    "n": entry["n"], "transport": transport, "threads": doc.get("threads_per_party", 1),
                    "bytes_per_key_without_base_ot": key["bytes_both"], "rounds_per_key_without_base_ot": key["rounds"],
                    "seconds_per_key_without_base_ot": key["seconds_median"],
                    "seconds_per_key_without_base_ot_min": key["seconds_min"],
                    "seconds_per_key_without_base_ot_max": key["seconds_max"],
                    "bytes_session": key["bytes_both"] + base["bytes_both"], "rounds_session": key["rounds"] + base["rounds"],
                    "seconds_session": statistics.median(session), "seconds_session_min": min(session),
                    "seconds_session_max": max(session),
                    "bytes_protocol_messages": rec["protocol_messages"]["bytes_both"],
                    "bytes_authenticated_bits": rec.get("authenticated_bits", {}).get("bytes_both"),
                    "bytes_session_per_party_max": max(party), "bytes_session_p0": party[0], "bytes_session_p1": party[1],
                    "bytes_messages_per_party_max": max(msgs), "bytes_correlations_per_party_max": max(abits),
                    "phases_sum_to_session": sum(party) == key["bytes_both"] + base["bytes_both"],
                    "repetitions": rec["repetitions"], "verified": bool(rec["all_outputs_verified"]),
                    "counts_match_formulas": bool(rec["counts_match_formulas"])})
    return {"status": "ok", "file": "raw/baselines_cpp.json",
            "note": "our re-implementations of the one-bit-leaky protocol of Zhang et al. and of semi-honest "
                    "Doerner-shelat-style generation on the same OT extension (bench_baselines.cpp); one fresh session "
                    "(base OT) and one key per repetition; session = base OT + key",
            "outputs": {"this_work": "a DPF key pair (payload 1) whose evaluation gives SPDZ2k sharings: values and "
                                     "tags over Z_{2^(k+s)} (MAC key Delta); k = s = 64",
                        "leaky_zhang": "shares of the whole truth table (2^n entries), a full-domain authenticated unit "
                                       "vector with GF(2^128) payload, one bit of the index leaks to a cheating party",
                        "semihonest_ds": "shares of the whole truth table with GF(2^128) payload, no MACs, no check"},
            "rows": rows}


def scrub_paths(text: str) -> str:
    """The text with the absolute location of the artifact replaced by 'artifact' (records carry no local paths)."""
    for root in {ARTIFACT, ARTIFACT.replace("\\", "/"), ARTIFACT.replace("/", "\\")}:
        text = text.replace(root, "artifact")
    return text


def group_attacks() -> dict:
    out = {}
    for key, cmd, path in (
            ("core", [sys.executable, "-m", "couplet.attacks.run_all", "--jobs", "4", "--out",
                      os.path.join(RAW, "attacks.json")], os.path.join(RAW, "attacks.json")),
            ("ppml", [sys.executable, os.path.join(_HERE, "ppml_attacks.py"), "--jobs", "4", "--out",
                      os.path.join(RAW, "attacks_ppml.json")], os.path.join(RAW, "attacks_ppml.json"))):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PY)
        doc = _read_json(path)
        out[key] = {"exit": proc.returncode, "seconds": round(time.perf_counter() - t0, 1),
                    "file": "raw/" + os.path.basename(path), "all_ok": bool(doc and doc.get("all_ok")),
                    "tail": scrub_paths((proc.stdout or "")[-1500:])}
        print(f"  attacks {key}: exit {proc.returncode}, all_ok {out[key]['all_ok']}, {out[key]['seconds']} s", flush=True)
    ok = all(v["exit"] == 0 and v["all_ok"] for v in out.values())
    return {"status": "ok" if ok else "failed", "reason": "" if ok else "an attack row failed its expectation",
            "measured_utc": _utc_now(), "runs": out}


# --------------------------------------------------------------------------- #
# The summary
# --------------------------------------------------------------------------- #

METHOD = ("two processes on 127.0.0.1, real preprocessing (base OT, OT extension, authenticated bits, leaky ANDs, "
          "bucketing, triple checks), sigma = 40, k = s = 64, one thread per party unless config.threads says otherwise; "
          "every repetition with --verify (TEST-ONLY step after the measured part, not counted); repetitions: 9 for runs "
          "under 10 s, 5 up to a minute, 3 above (the first run decides); median, minimum and maximum over the runs that "
          "are valid for timing; a run is valid for timing if the REST of the machine used at most "
          "{q:g} percent of the processor time of all logical processors while it lasted ({qw:g} percent for the rows on "
          "an emulated wide-area link, whose time is dominated by the emulated transfers and delays; own process tree "
          "subtracted through a job object; timing_threshold_percent of each row); a configuration is accepted only if every run passes the checks of the header of "
          "bench/run_campaign.py and all runs agree on bytes and rounds, in total and per phase; the inference runs "
          "use the test input (repetition mod 16) of the exported vectors, so a row covers as many test inputs as it "
          "has runs; bytes are payload bytes (the 4-byte framing and the 8-byte emulation stamp are not counted); "
          "seconds are wall-clock seconds of the slower party with both processes on one machine")

SCHEMA_NOTE = {
    "units": "groups u and k: one Pi_Gen session (session setup + ONE execution of the item's circuit with `batch` "
             "instances); and_gates = triples of the session (setup circuit 64 + n_and * batch); bytes in both "
             "directions; bytes_per_instance = bytes / batch",
    "e2e": "groups e, o, p, w: one inference session (manifest, setup with the preprocessing, generation of every item "
           "of the plan, online phase, release); prep_* = everything before the online phase (manifest exchange, "
           "preprocessing, generation circuits), online_* = the online phase (first-layer exchange, truncation openings, "
           "plane evaluation, table reads, checks, release); *_p0 / *_p1 = bytes SENT by that party; seconds of the "
           "slower party",
    "link": "lan: no emulation; wan20 / wan80: one-way delay 10 / 40 ms (round-trip 20 / 80 ms) and a token-bucket "
            "limit of 100 Mbit/s per direction in the channel of both processes (couplet/netem.hpp; depth 1,500 "
            "bytes; no message changes)",
    "status": "ok, failed (with reason), infeasible (with reason and estimate) or not_run (with reason and, where "
              "it can be computed, an estimate); rows that are not ok carry no measurement",
}


def _fmt_mb(b) -> str:
    return f"{b / 1e6:,.3f}"


def headline(s: dict) -> str:
    L = []
    L.append("Unit costs (two processes, real preprocessing; MB = 10^6 payload bytes, both directions)")
    head = f"{'configuration':<52} {'AND/inst':>9} {'batch':>5} {'MB/inst':>9} {'rounds':>6} {'s':>8} {'s/inst':>8} {'B':>2}"
    L += [head, "-" * len(head)]
    for r in s["units"] + s["keygen"]["rows"]:
        if r["status"] != "ok":
            L.append(f"{r['name']:<52} {r['status']}: {r.get('reason', '')}")
            continue
        L.append(f"{r['name']:<52} {r['n_and']:>9,} {r['batch']:>5} {r['bytes_per_instance'] / 1e6:>9.2f} {r['rounds']:>6} "
                 f"{r['seconds']:>8.3f} {r['seconds_per_instance']:>8.4f} {r['bucket_size']:>2}")
    L += ["", "Key-generation baselines (bench_baselines, tcp_two_processes): bytes and rounds of a session (base OT + key)"]
    for r in s["keygen"].get("baselines", []):
        if r["transport"] == "tcp_two_processes":
            L.append(f"  {r['protocol']:<14} n = {r['n']:>2}: {r['bytes_session']:>14,} bytes, {r['rounds_session']:>4} rounds, "
                     f"{r['seconds_session']:.3f} s")
    L += ["", "End-to-end inference (prep = everything before the online phase)"]
    head = (f"{'configuration':<58} {'sites':>5} {'AND gates':>12} {'prep MB/p':>10} {'prep s':>8} {'on rnd':>6} "
            f"{'on B':>7} {'on s':>7} {'total s':>8} {'exact':>5} {'dev':>4}")
    L += [head, "-" * len(head)]
    for r in s["e2e"]:
        if r["status"] != "ok":
            L.append(f"{r['name']:<58} {r['status']}: {r.get('reason', '')[:150]}")
            continue
        L.append(f"{r['name']:<58} {r['sites']:>5} {r['and_gates']:>12,} {r['prep_bytes_per_party_max'] / 1e6:>10.1f} "
                 f"{r['prep_seconds']:>8.2f} {r['online_rounds']:>6} {r['online_bytes']:>7,} {r['online_seconds']:>7.3f} "
                 f"{r['seconds']:>8.2f} {r['bit_exact_runs']:>2}/{r['runs_checked']:<2} "
                 f"{r['max_abs_deviation_from_exact_mirror_ulp']:>4}")
    L += ["", "Online layer (full MLP minus one hidden layer; per layer of 128 sites)"]
    for r in s["online_layers"]:
        if r["status"] != "ok":
            L.append(f"  ({r['n']}, {r['f']}) {r['mode']}: {r['status']}")
            continue
        L.append(f"  ({r['n']}, {r['f']}) {r['mode']:<5}: {r['layer_bytes']:>6,} bytes, {r['layer_rounds']} rounds, "
                 f"{r['layer_seconds'] * 1e3:8.2f} ms; plane eval {r['plane_eval_seconds_per_site'] * 1e3:.3f} ms/site, "
                 f"table read {r['table_read_seconds_per_site'] * 1e6:.1f} us/site")
    L += ["", "Accuracy on the synthetic test set (float / mirror exact / mirror round; agreement with the float model)"]
    for r in s["accuracy"]:
        L.append(f"  {r['name']:<16} {r['float_accuracy']:.4f} / {r['mirror_exact_accuracy']:.4f} / "
                 f"{r['mirror_round_accuracy']:.4f}; agreement {r['prediction_agreement']:.4f} / "
                 f"{r['prediction_agreement_round']:.4f}; overflows {r['code_overflows']} ({r['test_inputs_evaluated']} inputs)")
    if s.get("attacks"):
        L += ["", f"Attacks: {s['attacks'].get('status')} {s['attacks'].get('reason', '')}"]
    if s["not_measured"]:
        L += ["", "Not measured"]
        for r in s["not_measured"]:
            L.append(f"  {r['id']}: {r['status']}: {r.get('reason', '')}")
    return "\n".join(L) + "\n"


def build_summary(out_dir: str, items: list, tools: dict, machine: dict, extra_docs: list) -> dict:
    views = {"unit": [], "keygen": [], "e2e": []}
    for it in items:
        doc = _read_json(os.path.join(out_dir, it["name"] + ".json"))
        if doc is None:
            continue
        doc["groups"] = sorted(set(doc.get("groups", [])) | set(it["groups"]))
        views[it["view"]].append(build_row(doc))
    for doc in extra_docs:
        views[doc["view"]].append(build_row(doc))
    e2e = views["e2e"]
    accuracy = (tools.get("models") or {}).get("rows", [])
    acc_by = {r["name"]: r for r in accuracy}
    for r in e2e:
        a = acc_by.get(r["config"]["model"])
        if r["status"] == "ok" and a:
            r["accuracy"] = {k: a[k] for k in ("float_accuracy", "mirror_exact_accuracy", "mirror_round_accuracy",
                                               "prediction_agreement", "prediction_agreement_round",
                                               "test_inputs_evaluated", "code_overflows")}
    all_rows = views["unit"] + views["keygen"] + e2e
    not_ok = [{"id": r["id"], "status": r["status"], "reason": r.get("reason", ""), "estimate": r.get("estimate")}
              for r in all_rows if r["status"] != "ok"]
    ok_rows = [r for r in all_rows if r["status"] == "ok"]
    checks = {
        "configurations": len(all_rows), "accepted": len(ok_rows),
        "phase_bytes_sum_to_totals": all(r["phase_bytes_sum_ok"] and r["phase_rounds_sum_ok"] for r in ok_rows),
        "every_run_verified": all(r["verified"] for r in ok_rows),
        "inference_runs_bit_exact": all(r["bit_exact_runs"] == r["runs_checked"] for r in ok_rows if "bit_exact_runs" in r),
        "one_triple_request_unless_chunked": all(r["triple_requests"] == 1 for r in ok_rows if not r.get("chunk_and")),
        "not_ok": [x["id"] for x in not_ok],
    }
    summary = {
        "schema": SCHEMA, "generated_by": "python bench/run_campaign.py", "generated_utc": _utc_now(),
        "method": METHOD.format(q=QUIET["percent"], qw=QUIET["percent_wan"]), "schema_note": SCHEMA_NOTE, "machine": machine,
        "datasets": {"public_dataset_used": False,
                     "note": "all data are synthetic: the seeded task of couplet.tools.train_synthetic (make_dataset); "
                             "no public dataset was downloaded or used; the MNIST shapes (28 x 28 inputs, 784 features, "
                             "10 classes) are those of the synthetic image and flat tasks",
                     "training": TRAIN},
        "parameters": {"k": K, "s": S, "sigma": SIGMA, "formats": [list(x) for x in FORMATS],
                       "online_formats": [list(x) for x in ONLINE_FORMATS], "unit_formats": [list(x) for x in UNIT_NF],
                       "links": {k: {"delay_ms_one_way": v[0], "rtt_ms": 2 * v[0], "bandwidth_mbps": v[1]}
                                 for k, v in LINKS.items()},
                       "chunk_and": CHUNK_AND, "chunk_and_private": CHUNK_AND_PRIVATE,
                       "private_full": list(PRIVATE_FULL), "inputs_per_model": N_INPUTS},
        "checks": checks, "units": views["unit"],
        "keygen": {"rows": views["keygen"], "baselines": (tools.get("baselines") or {}).get("rows", []),
                   "baselines_status": (tools.get("baselines") or {}).get("status"),
                   "outputs": (tools.get("baselines") or {}).get("outputs")},
        "local": tools.get("local"), "e2e": e2e, "online_layers": derive_online_layers(e2e), "accuracy": accuracy,
        "attacks": _scrubbed(tools.get("attacks")), "not_measured": not_ok,
    }
    if QUICK["on"]:
        summary["quick"] = ("smoke run of bench/run_campaign.py --quick: a few small configurations with one "
                            "repetition each (see quick_keep); not the measurement campaign")
    return summary


def _scrubbed(attacks):
    """Group a of tools.json with local paths removed from the program output tails."""
    if not attacks:
        return attacks
    out = dict(attacks)
    out["runs"] = {k: dict(v, tail=scrub_paths(v.get("tail", ""))) for k, v in (attacks.get("runs") or {}).items()}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def prerequisites(groups: list) -> None:
    """Circuits of the unit rows that no plan exports (with --quick: every circuit its configurations load)."""
    if QUICK["on"]:
        names = list(QUICK_CORE_CIRCUITS) + [it["cfg"]["item"] for it in plan() if it["cfg"]["driver"] == "couplet_item"]
        subprocess.run([sys.executable, os.path.join(_HERE, "export_circuits.py"), "--skip-existing"] + names,
                       check=True, cwd=_PY)
        return
    if "u" in groups:
        names = [f"mattriple_16x16_k{K}_s{S}", f"triples1_k{K}_s{S}", f"inmask_p0_c1_k{K}_s{S}", f"dcf_n14_k{K}_s{S}_xor_one"]
        names += [f"site_n{n}_f{f}_k{K}_s{S}_{m}" for n, f in UNIT_NF for m in MODES]
        subprocess.run([sys.executable, os.path.join(_HERE, "export_circuits.py"), "--skip-existing"] + names,
                       check=True, cwd=_PY)


def main(argv=None) -> int:
    global RAW
    ap = argparse.ArgumentParser(description="measurement campaign of the Couplet evaluation")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: groups m, u, k, l, e, w on a few small configurations (quick_keep), one repetition, "
                         "output in experiments/quick (minutes instead of the campaign's hours)")
    ap.add_argument("--out", default=None, help="output directory (default: experiments/raw/campaign, "
                                                "with --quick experiments/quick/campaign)")
    ap.add_argument("--build-dir", default=os.path.join(CPP, "build"))
    ap.add_argument("--only", default="", help="comma-separated groups (m, u, k, l, e, o, p, w, a); default: all")
    ap.add_argument("--match", default="", help="only two-process configurations whose name contains this text")
    ap.add_argument("--names", default="", help="only these comma-separated configuration names (exact)")
    ap.add_argument("--repeat", type=int, default=None,
                    help="repetitions of runs under 10 s (5 up to a minute, 3 above); default 9, with --quick 1")
    ap.add_argument("--quiet-percent", type=float, default=QUIET["percent"])
    ap.add_argument("--quiet-percent-wan", type=float, default=QUIET["percent_wan"],
                    help="the same for the rows on an emulated wide-area link")
    ap.add_argument("--quiet-wait", type=float, default=None,
                    help=f"longest wait for a quiet machine before a run, seconds (default {QUIET['max_wait']:g}, "
                         "with --quick 30)")
    ap.add_argument("--wan-max-run", type=float, default=1200.0, help="longest estimated run of a wide-area row, seconds")
    ap.add_argument("--force", action="store_true", help="run again what the output directory already holds")
    ap.add_argument("--timeout", type=float, default=7200.0, help="seconds per run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    QUICK["on"] = bool(args.quick)
    if args.quick:
        RAW = os.path.join(ARTIFACT, "experiments", "quick")
    if args.out is None:
        args.out = os.path.join(RAW, "campaign") if args.quick else DEFAULT_OUT
    if args.repeat is None:
        args.repeat = 1 if args.quick else 9
    if args.quiet_wait is None:
        args.quiet_wait = 30.0 if args.quick else QUIET["max_wait"]
    QUIET["percent"], QUIET["max_wait"] = args.quiet_percent, args.quiet_wait
    QUIET["percent_wan"] = args.quiet_percent_wan
    order = list("mukleopwa")
    groups = [g.strip() for g in args.only.split(",") if g.strip()] or order
    if args.quick:
        groups = [g for g in groups if g in QUICK_GROUPS]
    items = plan()
    if args.list:
        for it in items:
            print(f"{','.join(it['groups']):<6} {it['view']:<7} {it['name']}")
        return 0
    os.makedirs(args.out, exist_ok=True)
    tools_path = os.path.join(args.out, "tools.json")
    tools = _read_json(tools_path) or {}

    def finish() -> dict:
        compiler = ""
        for it in items:
            rec = _read_json(os.path.join(args.out, it["name"] + "_p0.json"))
            if rec:
                compiler = format_msvc(rec.get("compiler", ""))
                break
        machine = _read_json(os.path.join(RAW, "machine.json"))
        if machine is None or (compiler and machine.get("compiler") != compiler):
            machine = machine_info(compiler)
            _write_json(os.path.join(RAW, "machine.json"), machine)
        extra = [] if QUICK["on"] else private_full_assessment(args.out)
        if not QUICK["on"]:
            _write_json(os.path.join(args.out, "private_full.json"),
                        {"schema": "couplet.m3.campaign.private_full/2", "generated_utc": _utc_now(),
                         "note": "not-run documents of the PRIVATE_FULL configurations without a file, rebuilt by every "
                                 "summary (private_full_assessment); empty when every such configuration has a file",
                         "docs": extra})
        summary = build_summary(args.out, items, tools, machine, extra)
        _write_json(os.path.join(args.out, "summary.json"), summary)
        with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(headline(summary))
        return summary

    if args.summary_only:
        raw = _read_json(os.path.join(RAW, "baselines_cpp.json"))
        if raw and (tools.get("baselines") or {}).get("status") == "ok":
            tools["baselines"].update(baseline_summary(raw))       # derived fields only
            _write_json(tools_path, tools)
        print(headline(finish()))
        return 0
    binaries = {d: _binary(d, args.build_dir) for d in ("couplet_item", "couplet_party", "couplet_infer")}
    wanted = set(x for x in args.names.split(",") if x)
    ran = set()
    prerequisites(groups)
    for g in order:
        if g not in groups:
            continue
        if g == "m" and not (args.match or args.names):
            print("group m: models ...", flush=True)
            tools["models"] = group_models(args.force)
            _write_json(tools_path, tools)
            print(f"group m: {tools['models']['status']} {tools['models'].get('reason', '')}", flush=True)
            continue
        if g == "l" and not (args.match or args.names):
            if args.force or (tools.get("local") or {}).get("status") != "ok":
                print("group l: bench_ppml ...", flush=True)
                tools["local"] = group_local(args.build_dir)
                _write_json(tools_path, tools)
        if g == "k" and not (args.match or args.names):
            if args.force or (tools.get("baselines") or {}).get("status") != "ok":
                print("group k: bench_baselines ...", flush=True)
                tools["baselines"] = group_baselines(args.build_dir)
                _write_json(tools_path, tools)
        if g == "a" and not (args.match or args.names):
            if args.force or (tools.get("attacks") or {}).get("status") != "ok":
                print("group a: attacks ...", flush=True)
                tools["attacks"] = group_attacks()
                _write_json(tools_path, tools)
            continue
        for it in items:
            if g not in it["groups"] or args.match not in it["name"] or (args.names and it["name"] not in wanted):
                continue
            if it["name"] in ran:                 # a configuration of several groups runs once per invocation
                continue
            ran.add(it["name"])
            path = os.path.join(args.out, it["name"] + ".json")
            old = _read_json(path)
            if old is not None and not args.force and old.get("status") in ("ok", "infeasible"):
                continue
            if it["cfg"]["link"] != "lan":
                verdict = estimate_wan(args.out, it, args.wan_max_run, args.repeat)
                if verdict is not None:
                    _write_json(path, verdict)
                    print(f"{verdict['status'].upper()} {it['name']}: {verdict['reason']}", flush=True)
                    continue
            print(f"[{g}] {it['name']}", flush=True)
            run_config(binaries, it, args.out, args.repeat, args.timeout)
            finish()
    summary = finish()
    print()
    print(headline(summary))
    bad = [x for x in summary["checks"]["not_ok"] if not any(
        x == r["id"] and (r["status"] == "infeasible" or (r["status"] == "not_run" and r.get("estimate")))
        for r in summary["not_measured"])]
    bad += [k for k in ("local", "baselines", "models", "attacks") if k in tools and tools[k].get("status") != "ok"]
    if bad:
        print("NOT OK:", ", ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
