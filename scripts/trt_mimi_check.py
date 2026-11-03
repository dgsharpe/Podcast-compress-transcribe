"""Mimi's convolutional encoder (SEANet: most of Mimi's GPU time) as a TensorRT engine vs eager PyTorch (fp16 weights
under autocast, as the hub runs it): time and output difference, 32 streams x 20 s, not streamed.

    python scripts/trt_mimi_check.py [batch] [seconds]
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def timed(f, reps=10):
    import torch
    f()
    f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps, out


def main():
    import tensorrt as trt
    import torch
    import mimi_file
    from hub.autocast_weights import half_weights

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    samples = int(secs * 12.5) * mimi_file.SPF
    res = {"device": torch.cuda.get_device_name(0), "tensorrt": trt.__version__, "batch": B, "seconds": secs}
    model = mimi_file.load_model("cuda")
    half_weights(model.encoder)
    seanet = model.encoder.encoder_seanet.eval()
    x = torch.randn(B, 1, samples, device="cuda") * 0.1

    def eager():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return seanet(x)
    t_eager, ref = timed(eager)
    res["eager_ms"] = round(1000 * t_eager, 2)

    # all-fp16 copy for export (TensorRT builds the fp16 kernels from it)
    work = Path(tempfile.mkdtemp(prefix="trt-mimi-"))
    onnx_path = work / "seanet.onnx"
    import copy
    m16 = copy.deepcopy(seanet).half().eval()
    t = time.perf_counter()
    example = (x.half().contiguous(),)
    with torch.no_grad():
        try:  # the TorchScript exporter (static shapes; still there, deprecated), else torch.export's
            torch.onnx.export(m16, example, str(onnx_path), input_names=["audio"], output_names=["latent"],
                              opset_version=18, dynamo=False)
        except Exception as e:  # noqa: BLE001
            res["export_fallback"] = f"{type(e).__name__}: {str(e)[:150]}"
            torch.onnx.export(m16, example, str(onnx_path), input_names=["audio"], output_names=["latent"],
                              opset_version=18, dynamo=True)
    res["export_s"] = round(time.perf_counter() - t, 1)
    del m16

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("; ".join(str(parser.get_error(k)) for k in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    res["build_s"] = round(time.perf_counter() - t, 1)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    xin = x.half().contiguous()
    out_shape = tuple(engine.get_tensor_shape("latent"))
    out = torch.empty(out_shape, dtype=torch.float16, device="cuda")
    ctx.set_tensor_address("audio", xin.data_ptr())
    ctx.set_tensor_address("latent", out.data_ptr())
    stream = torch.cuda.current_stream()

    def run_trt():
        ctx.execute_async_v3(stream.cuda_stream)
        return out
    t_trt, got = timed(run_trt)
    res["trt_ms"] = round(1000 * t_trt, 2)
    res["speedup"] = round(t_eager / t_trt, 2)
    a, b = got.float(), ref.float()
    res["shape_match"] = tuple(a.shape) == tuple(b.shape)
    if res["shape_match"]:
        res["max_rel_diff"] = round(float((a - b).abs().max() / b.abs().max()), 5)
        res["mean_rel_diff"] = round(float((a - b).abs().mean() / b.abs().mean()), 5)
    res["x_rt_eager"] = round(B * secs / t_eager)
    res["x_rt_trt"] = round(B * secs / t_trt)
    print("TRTCHECK " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
