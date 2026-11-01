"""Run in a throwaway NeMo venv (data/venv-nemo: CPU torch + nemo_toolkit[asr]): NVIDIA reference for the hub's
Parakeet encoders, on batches exported to data/nemo_check/ from the hub env."""
import glob, json, difflib, os, re
import numpy as np, torch
def norm(t):
    return re.sub(r"[^\w' ]", " ", t.lower()).split()
def main():
    import nemo.collections.asr as nemo_asr
    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v2/snapshots/*/parakeet-tdt-0.6b-v2.nemo"))[0]
    model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu").eval()
    torch.set_num_threads(16)
    d = np.load("data/nemo_check/segments.npz"); meta = json.load(open("data/nemo_check/segments.json"))
    audio = [d[f"s{i}"] for i in range(len(meta["bounds"]))]
    hyps = model.transcribe(audio, batch_size=8, verbose=False)
    ref = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    R = [w for t in ref for w in norm(t)]
    for name in ("onnx2torch", "native"):
        H = [w for t in meta[name] for w in norm(t)]
        same = sum(norm(a) == norm(b) for a, b in zip(ref, meta[name]))
        sm = difflib.SequenceMatcher(a=R, b=H, autojunk=False)
        errs = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
        print(f"{name:10s} vs NeMo: segments identical {same}/{len(ref)}, word errors {errs}/{len(R)} ({100 * errs / len(R):.2f}%)", flush=True)
    for a, b, c in zip(ref, meta["onnx2torch"], meta["native"]):
        if norm(a) != norm(b) or norm(a) != norm(c):
            print(" NeMo :", a[:130]); print(" onnx :", b[:130]); print(" native:", c[:130]); break
if __name__ == "__main__":
    main()
