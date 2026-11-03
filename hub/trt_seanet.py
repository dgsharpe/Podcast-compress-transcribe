"""Mimi's convolutional encoder (SEANet, most of Mimi's GPU time) as a TensorRT engine for hub/mimi_static.py's
streaming step (StaticMimiEncoder(trt=True)): on an H100 the whole Mimi step 38.3 -> 29.7 ms a chunk (16 x 20 s),
its codes decoding to audio as close to the original as before (log-mel L1 0.585 either way; 80% of the codes the
same, 96% of the first 8 codebooks'; scripts/mimi_static_check.py --trt).

Streaming needs each causal conv's left padding from the chunk before; here those are the engine's inputs, and
their next values its outputs: a step is (audio, states...) -> (latent, states...). One engine per (batch, chunk
length), built from an fp16 ONNX export on first use (~3 min) and cached beside the AOT packages (hub/aoti.py) per
TensorRT version and GPU. TensorRT's Python bindings crash free-threaded Python (hub/bulk.py's), so the engine
runs through a small C shim (compiled on first use against TensorRT's headers, libnvinfer-headers-dev, and the
pip package's libnvinfer) loaded with ctypes; building needs the bindings, so on free-threaded Python it runs in
MIMI_HUB_TRT_BUILD_PYTHON (default /usr/local/bin/python).
"""
import ctypes
import glob
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

VERSION = 2

_SHIM = r"""
#include <NvInferRuntime.h>
#include <cstdio>
#include <fstream>
#include <vector>
using namespace nvinfer1;
namespace {
struct Logger : ILogger {
    void log(Severity s, const char* m) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "trt: %s\n", m);
    }
} logger;
struct Handle { IRuntime* rt; ICudaEngine* engine; IExecutionContext* ctx; };
}
extern "C" {
void* mt_load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> plan((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    IRuntime* rt = createInferRuntime(logger);
    if (!rt) return nullptr;
    ICudaEngine* e = rt->deserializeCudaEngine(plan.data(), plan.size());
    if (!e) return nullptr;
    IExecutionContext* c = e->createExecutionContext();
    if (!c) return nullptr;
    return new Handle{rt, e, c};
}
int mt_set(void* h, const char* name, void* ptr) {
    return static_cast<Handle*>(h)->ctx->setTensorAddress(name, ptr) ? 0 : 1;
}
int mt_enqueue(void* h, void* stream) {
    return static_cast<Handle*>(h)->ctx->enqueueV3(static_cast<cudaStream_t>(stream)) ? 0 : 1;
}
int mt_shape(void* h, const char* name, long long* dims, int max_dims) {
    Dims d = static_cast<Handle*>(h)->engine->getTensorShape(name);
    for (int i = 0; i < d.nbDims && i < max_dims; ++i) dims[i] = d.d[i];
    return d.nbDims;
}
}
"""


def _libdir():
    """The directory with the pip TensorRT's libnvinfer (this Python's site-packages, else the build Python's)."""
    cands = [p for d in sys.path if (p := os.path.join(d, "tensorrt_libs")) and os.path.exists(p)]
    cands += glob.glob("/usr/local/lib/python3*/site-packages/tensorrt_libs")
    for d in cands:
        if glob.glob(os.path.join(d, "libnvinfer.so.*")):
            return d
    raise RuntimeError("TensorRT's libnvinfer not found (pip install tensorrt-cu13)")


def _trt_version():
    d = Path(_libdir()).parent
    for info in sorted(d.glob("tensorrt_cu*_libs-*.dist-info")) + sorted(d.glob("tensorrt_libs-*.dist-info")):
        return info.name.split("-")[1]
    return "unknown"


_shim = None


def _lib():
    """The shim, compiled once per TensorRT version and loaded."""
    global _shim
    if _shim is not None:
        return _shim
    from hub import aoti
    libdir = _libdir()
    so_name = os.path.basename(glob.glob(os.path.join(libdir, "libnvinfer.so.*"))[0])
    key = hashlib.sha256(f"{_SHIM} {_trt_version()} {so_name}".encode()).hexdigest()[:12]
    so = aoti.DIR / f"mimi-trt-shim-{key}.so"
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        out = so.with_suffix(".building.so")  # beside it (the cache may be another filesystem than /tmp)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "shim.cpp")
            Path(src).write_text(_SHIM)
            subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-std=c++17", src, "-o", str(out),
                            "-I/usr/include/x86_64-linux-gnu", f"-I{cuda}/include", f"-L{libdir}",
                            f"-l:{so_name}", f"-Wl,-rpath,{libdir}"], check=True, capture_output=True, text=True)
        out.replace(so)
    ctypes.CDLL(os.path.join(libdir, so_name), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(so))
    lib.mt_load.restype = ctypes.c_void_p
    lib.mt_load.argtypes = [ctypes.c_char_p]
    lib.mt_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    lib.mt_enqueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mt_shape.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
    _shim = lib
    return lib


class _StateCache:
    """Stands in for _PadCache while exporting: the left padding of the k-th conv that has one (layers[k]) is
    states[k], its next one goes to out[k]; convs without (kernel 1) get an empty tensor."""

    def __init__(self, layers, states):
        self.slot = {layer: k for k, layer in enumerate(layers)}
        self.states, self.out = list(states), [None] * len(states)

    def update(self, hidden_states, layer_idx):
        k = self.slot.get(layer_idx)
        if k is None:
            return hidden_states[..., :0]
        prev = self.states[k]
        self.out[k] = hidden_states[..., hidden_states.shape[-1] - prev.shape[-1]:]
        return prev


class _Step(torch.nn.Module):
    def __init__(self, seanet, layers):
        super().__init__()
        self.seanet, self.layers = seanet, layers

    def forward(self, x, *states):
        cache = _StateCache(self.layers, states)
        h = self.seanet(x, padding_cache=cache)
        return (h, *cache.out)


def state_shapes(seanet, batch, samples, device):
    """[(layer index, channels, pad)] of each conv with a left-padding state, from one eager step."""
    names = list(seanet._conv1d_layer_names)
    pads = [int(seanet.get_submodule(n).padding_total) for n in names]
    shapes = {}

    class Probe:
        def update(self, h, idx):
            shapes[idx] = (h.shape[1], pads[idx])
            return h.new_zeros((*h.shape[:2], pads[idx]))
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
        seanet(torch.zeros(batch, 1, samples, device=device), padding_cache=Probe())
    return [(i, *shapes[i]) for i in range(len(pads)) if pads[i] > 0]


def build(seanet, batch, samples, path):
    """The engine for (batch, 1, samples) fp16 audio, written to [path] (needs TensorRT's Python bindings)."""
    import copy
    import tensorrt as trt
    device = next(seanet.parameters()).device
    shapes = state_shapes(seanet, batch, samples, device)
    n = len(shapes)
    step = _Step(copy.deepcopy(seanet).half().eval(), [i for i, _, _ in shapes])
    x = torch.zeros(batch, 1, samples, device=device, dtype=torch.float16)
    states = [torch.zeros(batch, c, p, device=device, dtype=torch.float16) for _, c, p in shapes]
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "seanet.onnx")
        with torch.no_grad():
            torch.onnx.export(step, (x, *states), onnx_path, opset_version=18, dynamo=False,
                              input_names=["audio", *[f"state{k}" for k in range(n)]],
                              output_names=["latent", *[f"next{k}" for k in range(n)]])
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError("; ".join(str(parser.get_error(k)) for k in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT build failed")
    tmp_path = Path(path).with_suffix(".building")
    tmp_path.write_bytes(bytes(plan))
    tmp_path.replace(path)


def _build_elsewhere(batch, samples, path):
    """Build in a Python with GIL (TensorRT's bindings crash a free-threaded one): Mimi loaded there."""
    py = os.environ.get("MIMI_HUB_TRT_BUILD_PYTHON", "/usr/local/bin/python")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHON_GIL", "PYTHONPATH")}
    root = Path(__file__).resolve().parent.parent
    subprocess.run([py, "-m", "hub.trt_seanet", "build", str(batch), str(samples), str(path)], cwd=root, env=env,
                   check=True)


class Engine:
    """step(x) for (batch, 1, samples) audio; the left-padding states (fp16) advance in place."""

    def __init__(self, seanet, batch, samples, device):
        self.shapes = state_shapes(seanet, batch, samples, device)
        key = hashlib.sha256(f"{VERSION} {_trt_version()} {torch.cuda.get_device_name(device)} {batch} {samples} "
                             f"{self.shapes}".encode()).hexdigest()[:16]
        from hub import aoti
        path = aoti.DIR / f"mimi-seanet-{batch}x{samples}-{key}.plan"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if getattr(sys, "_is_gil_enabled", lambda: True)():
                build(seanet, batch, samples, path)
            else:
                _build_elsewhere(batch, samples, path)
        lib = _lib()
        self.lib, self.h = lib, lib.mt_load(str(path).encode())
        if not self.h:
            raise RuntimeError(f"TensorRT couldn't load {path}")
        dims = (ctypes.c_longlong * 8)()
        nd = lib.mt_shape(self.h, b"latent", dims, 8)
        with torch.inference_mode():
            self.x = torch.zeros(batch, 1, samples, device=device, dtype=torch.float16)
            self.states = [torch.zeros(batch, c, p, device=device, dtype=torch.float16) for _, c, p in self.shapes]
            self.next = [torch.zeros_like(s) for s in self.states]
            self.out = torch.empty(tuple(dims[:nd]), device=device, dtype=torch.float16)
        bind = [(b"audio", self.x), (b"latent", self.out)]
        bind += [(f"state{k}".encode(), s) for k, s in enumerate(self.states)]
        bind += [(f"next{k}".encode(), s) for k, s in enumerate(self.next)]
        for name, t in bind:
            if lib.mt_set(self.h, name, ctypes.c_void_p(t.data_ptr())):
                raise RuntimeError(f"TensorRT: no tensor {name.decode()}")

    def step(self, x):
        """One chunk: x (batch, 1, samples) float audio -> the latent (fp16); the states advance."""
        self.x.copy_(x)
        if self.lib.mt_enqueue(self.h, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)):
            raise RuntimeError("TensorRT enqueue failed")
        for s, nxt in zip(self.states, self.next, strict=True):
            s.copy_(nxt)
        return self.out

    def reset(self):
        for s in self.states:
            s.zero_()


if __name__ == "__main__":  # python -m hub.trt_seanet build BATCH SAMPLES PATH (in a Python with GIL)
    if sys.argv[1:2] == ["build"]:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import mimi_file
        from hub.autocast_weights import half_weights
        model = mimi_file.load_model("cuda")
        half_weights(model.encoder)
        build(model.encoder.encoder_seanet, int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
