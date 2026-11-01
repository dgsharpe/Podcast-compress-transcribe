"""Which shared component keeps the hub's Parakeet transcripts from NeMo's: features or decoder? (NeMo venv)

    data/venv-nemo/bin/python scripts/nemo_component_check.py

Run in the throwaway NeMo venv (data/venv-nemo: CPU torch + nemo_toolkit[asr] + onnx-asr[cpu]), on the VAD segments
exported to data/nemo_check/segments_full.{npz,json} from the hub env. onnx_asr's pipeline is re-run with parts
swapped for NeMo's (its features = model.preprocessor, its encoder = model.encoder; native == NeMo's encoder):
  A  onnx_asr features + ONNX encoder + onnx_asr decoder     (the hub before today)
  B  onnx_asr features + NeMo encoder + onnx_asr decoder     (the hub now: native encoder)
  C  NeMo features     + NeMo encoder + onnx_asr decoder     (exact features)
  F  NeMo features     + NeMo encoder + NeMo TDT decoding    (same batches as A-C)
  D  NeMo end to end (model.transcribe)                       (reference)
Word differences are counted after lower-casing and dropping punctuation.
"""

import difflib
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BATCH = 8


def words(texts):
    return [w for t in texts for w in re.sub(r"[^\w' ]", " ", t.lower()).split()]


def diff(ref, hyp):
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def main():
    import nemo.collections.asr as nemo_asr
    import onnx_asr
    from onnx_asr.utils import pad_list

    torch.set_num_threads(16)
    path = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v2/snapshots/*/"
                         "parakeet-tdt-0.6b-v2.nemo"))[0]
    model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu").eval()
    base = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", str(ROOT / "data/nemo_check/parakeet_onnx"))
    asr = base.asr
    onnx_pre, onnx_enc = asr._preprocessor, asr._encode

    def nemo_pre(waveforms, lens):
        with torch.inference_mode():
            f, fl = model.preprocessor(input_signal=torch.from_numpy(waveforms), length=torch.from_numpy(lens))
        return f.numpy(), fl.numpy().astype(np.int64)

    def nemo_enc(features, lens):
        with torch.inference_mode():
            e, el = model.encoder(audio_signal=torch.from_numpy(features), length=torch.from_numpy(lens))
        return e.transpose(1, 2).numpy(), el.numpy().astype(np.int64)  # onnx_asr wants (B, T, D)

    d = np.load(ROOT / "data/nemo_check/segments_full.npz")
    meta = json.load(open(ROOT / "data/nemo_check/segments_full.json"))
    audio = [d[f"s{i}"].astype(np.float32) for i in range(len(meta["bounds"]))]
    if len(sys.argv) > 1:
        audio = audio[: int(sys.argv[1])]

    def run_onnx_asr(pre, enc):
        asr._preprocessor, asr._encode = pre, enc
        out = []
        for i in range(0, len(audio), BATCH):
            out += [r.text for r in asr.recognize_batch(*pad_list(audio[i:i + BATCH]))]
        asr._preprocessor, asr._encode = onnx_pre, onnx_enc
        return out

    res, texts = {}, {}
    t = time.perf_counter()
    hyps = model.transcribe(audio, batch_size=BATCH, verbose=False)
    texts["D"] = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    res["D_s"] = round(time.perf_counter() - t)
    model.eval()  # transcribe() leaves the model in training mode (dropout on), which wrecks B and C
    for name, pre, enc in (("A", onnx_pre, onnx_enc), ("B", onnx_pre, nemo_enc), ("C", nemo_pre, nemo_enc)):
        t = time.perf_counter()
        texts[name] = run_onnx_asr(pre, enc)
        res[f"{name}_s"] = round(time.perf_counter() - t)
    t = time.perf_counter()
    texts["F"] = []
    for i in range(0, len(audio), BATCH):
        w, wl = pad_list(audio[i:i + BATCH])
        f, fl = nemo_pre(w, wl)
        with torch.inference_mode():
            e, el = model.encoder(audio_signal=torch.from_numpy(f), length=torch.from_numpy(fl))
            hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        texts["F"] += [h.text if hasattr(h, "text") else str(h) for h in hyps]
    res["F_s"] = round(time.perf_counter() - t)
    ref = words(texts["D"])
    res["ref_words"] = len(ref)
    for name in "ABCF":
        e = diff(ref, words(texts[name]))
        res[name] = {"word_diffs": e, "pct": round(100 * e / len(ref), 2),
                     "segments_identical": sum(words([a]) == words([b]) for a, b in zip(texts["D"], texts[name]))}
        print(f"{name}: {e} word differences from NeMo ({res[name]['pct']}%), "
              f"{res[name]['segments_identical']}/{len(audio)} segments identical", flush=True)
    shown = 0
    for a, b, c in zip(texts["D"], texts["B"], texts["C"]):
        if words([a]) != words([b]) and shown < 4:
            print(f"  NeMo: {a[:120]}\n  B   : {b[:120]}\n  C   : {c[:120]}", flush=True)
            shown += 1
    json.dump({"summary": res, "texts": texts}, open(ROOT / "outputs/nemo_component_check.json", "w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
