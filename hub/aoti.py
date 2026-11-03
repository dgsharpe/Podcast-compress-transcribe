"""AOTInductor packages of the hub's models (MIMI_HUB_COMPILE=aot): torch.export + inductor ahead of time, loaded as
a C++ library. Unlike torch.compile there's no Python frame hook per call: on free-threaded Python torch.compile
halved hub/bulk.py's throughput, these run as any other op. The packages hold no weights (the loaded model uses the
module's own GPU tensors), are cached in MIMI_HUB_MODELS/aoti per torch version, GPU and input shapes, and built
once under a file lock (several hub processes may load at once).
"""
import fcntl
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DIR = Path(os.environ.get("MIMI_HUB_AOTI") or Path(os.environ.get("MIMI_HUB_MODELS", ROOT / "hub_models")) / "aoti")
VERSION = 1  # bump when what a package is built from changes (wrappers, configs)
# MIMI_HUB_AOTI_MARCH, e.g. x86-64-v3: build the packages' C++ for that CPU level instead of -march=native, so one
# package serves every host that has it (Modal's vary; otherwise each new CPU type rebuilds, minutes on a cold start).
# Inductor picks its vector instructions separately (AVX-512 where the building host has it: SIGILL on an AVX2 host),
# so with MARCH also MIMI_HUB_AOTI_SIMDLEN bits (256, AVX2, x86-64-v3's).
MARCH = os.environ.get("MIMI_HUB_AOTI_MARCH", "")
SIMDLEN = int(os.environ.get("MIMI_HUB_AOTI_SIMDLEN", "256"))


def _cpu():
    """The CPU, for the cache key: inductor builds a package's C++ with -march=native, so one built on another
    CPU type can die with SIGILL here (Modal's hosts vary). With MARCH, that level."""
    if MARCH:
        return f"march={MARCH} simdlen={SIMDLEN}"
    try:
        lines = Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines()
        return " ".join(ln.split(":", 1)[1].strip() for ln in lines if ln.split(":")[0].strip() in ("model name",
                                                                                                    "flags"))
    except OSError:
        import platform
        return platform.processor() or platform.machine()


def mode(device_type):
    """MIMI_HUB_COMPILE for a model on [device_type]: "aot" (these packages), "jit" (torch.compile) or None.
    auto: aot on CUDA / ROCm GPUs; 1 = aot. (jit on free-threaded Python halved hub/bulk.py's throughput.)"""
    m = os.environ.get("MIMI_HUB_COMPILE", "auto")
    if m == "auto":
        return "aot" if device_type == "cuda" else None
    return {"1": "aot", "aot": "aot", "jit": "jit"}.get(m)


def _toolchain():
    """What inductor's ahead-of-time build needs and doesn't find on its own: ROCM_HOME from the ROCm SDK's pip
    package (rocm[devel]), and an `openssl sha512` (it hashes a header with it; not on the server's PATH)."""
    if torch.version.hip and not os.environ.get("ROCM_HOME"):
        sdk = Path(sys.executable).parent / "rocm-sdk"
        if sdk.exists():
            os.environ["ROCM_HOME"] = subprocess.run([str(sdk), "path", "--root"], capture_output=True,
                                                     text=True).stdout.strip()
    if shutil.which("openssl") is None:
        shim = Path(tempfile.mkdtemp(prefix="mimi-openssl-"))
        (shim / "openssl").write_text(
            f"#!/bin/sh\n# openssl sha512 FILE, as inductor calls it\nexec {sys.executable} -c "
            "'import hashlib,sys; print(\"SHA2-512(%s)= %s\" % (sys.argv[2], "
            "hashlib.sha512(open(sys.argv[2], \"rb\").read()).hexdigest()))' \"$@\"\n")
        (shim / "openssl").chmod(0o755)
        os.environ["PATH"] = f"{shim}{os.pathsep}{os.environ.get('PATH', '')}"
    if MARCH:  # inductor always passes -march=native: a compiler wrapper that swaps it
        from torch._inductor import config
        cxx = shutil.which(os.environ.get("CXX", "g++"))
        shim = Path(tempfile.mkdtemp(prefix="mimi-cxx-")) / "g++"  # (inductor tells gcc from clang by --version)
        shim.write_text(f'#!/bin/sh\nfor a; do shift; [ "$a" = -march=native ] && a=-march={MARCH}; '
                        f'set -- "$@" "$a"; done\nexec {cxx} "$@"\n')
        shim.chmod(0o755)
        config.cpp.cxx = (None, str(shim))
        config.cpp.simdlen = SIMDLEN


def _no_exec_stack(package):
    """Clear the executable-stack flag of the package's libraries: on ROCm the wrapper .so comes out with an RWE
    GNU_STACK segment, which glibc refuses to dlopen ("cannot enable executable stack")."""
    tmp = package.with_suffix(".tmp")
    with zipfile.ZipFile(package) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename.endswith(".so") and data[:4] == b"\x7fELF" and data[4] == 2:  # ELF64
                data = bytearray(data)
                phoff, = struct.unpack_from("<Q", data, 0x20)
                phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
                for k in range(phnum):
                    off = phoff + k * phentsize
                    p_type, p_flags = struct.unpack_from("<II", data, off)
                    if p_type == 0x6474E551:  # PT_GNU_STACK
                        struct.pack_into("<I", data, off + 4, p_flags & ~1)  # drop PF_X
                data = bytes(data)
            dst.writestr(info, data)
    tmp.replace(package)


def load(name, module, args, dynamic_shapes, num_runners=2):
    """[module] (eval, on its GPU) as an AOTInductor model for inputs like [args] (dims in [dynamic_shapes]
    dynamic): built on first use, then loaded from the cache. Returns a callable: model(*args) -> its outputs."""
    from torch._inductor import aoti_compile_and_package
    from torch._inductor.package import load_package

    device = torch.cuda.get_device_name(args[0].device)
    spec = repr([(tuple(a.shape), str(a.dtype)) for a in args]) + repr(dynamic_shapes)
    key = hashlib.sha256(f"{VERSION} {name} {torch.__version__} {device} {_cpu()} {spec}".encode()).hexdigest()[:16]
    path = DIR / f"{name}-{key}.pt2"
    DIR.mkdir(parents=True, exist_ok=True)
    # locked locally (the processes building at once share a machine; the cache may be on a network volume)
    with open(Path(tempfile.gettempdir()) / f"mimi-aoti-{path.stem}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            _toolchain()
            with torch.no_grad():
                ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes)
            tmp = path.with_suffix(".building.pt2")
            aoti_compile_and_package(ep, package_path=str(tmp),
                                     inductor_configs={"aot_inductor.package_constants_in_so": False})
            if torch.version.hip:
                _no_exec_stack(tmp)
            tmp.replace(path)
    model = load_package(str(path), num_runners=num_runners)
    state = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    model.load_constants({n: state[n] for n in model.get_constant_fqns()}, check_full_update=True,
                         user_managed=True)
    return model
