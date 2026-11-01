"""Run in a throwaway NeMo venv (data/venv-nemo: CPU torch + nemo_toolkit[asr]): NVIDIA reference for the hub's
Parakeet encoders, on batches exported to data/nemo_check/ from the hub env."""
import glob, json
import numpy as np, torch
def stats(got, ref, lens):
    cos, rel = [], []
    for r in range(ref.shape[0]):
        k = int(lens[r]); a, b = got[r, :, :k], ref[r, :, :k]
        cos.append(float(((a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))).min()))
        rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    return min(cos), max(rel)
def main():
    import nemo.collections.asr as nemo_asr
    path = glob.glob("~/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v2/snapshots/*/parakeet-tdt-0.6b-v2.nemo")[0]
    model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu").eval()
    torch.set_num_threads(16)
    d = np.load("data/nemo_check/batches.npz")
    out = {"onnx2torch": [], "native": []}
    for b in range(4):
        x, n = d[f"x{b}"], d[f"n{b}"]
        with torch.inference_mode():
            enc, enc_len = model.encoder(audio_signal=torch.from_numpy(x), length=torch.from_numpy(n))
        ref, rl = enc.numpy(), enc_len.numpy()
        for name, key in (("onnx2torch", "onnx"), ("native", "native")):
            got = d[f"{key}{b}"]
            assert got.shape == ref.shape, (name, got.shape, ref.shape)
            out[name].append(stats(got, ref, rl))
        print(f"batch {b} {x.shape}: onnx2torch min-cos {out['onnx2torch'][-1][0]:.6f} max-rel-err {out['onnx2torch'][-1][1]:.2e} | "
              f"native min-cos {out['native'][-1][0]:.6f} max-rel-err {out['native'][-1][1]:.2e}", flush=True)
    print("SUMMARY " + json.dumps({k: {"min_cos": round(min(c for c, _ in v), 6), "max_rel_err": float(f"{max(r for _, r in v):.3g}")} for k, v in out.items()}))
if __name__ == "__main__":
    main()
