"""Download the hub's public models into the HF cache, so it can then run with HF_HUB_OFFLINE=1.

    python hub/prefetch.py

Mimi, Parakeet TDT's ONNX export (preprocessor, decoder; its encoder is replaced on a GPU) and Silero VAD. The
pyannote diarization model is gated: fetch it once with a token, or point the hub at a local copy. onnxruntime
rejects external weights that resolve outside the model's directory, as the cache's symlinks into blobs/ can, so
the ONNX snapshot's files are made real files.
"""
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    snapshot_download("kyutai/mimi")
    snap = Path(snapshot_download("istupakov/parakeet-tdt-0.6b-v2-onnx"))
    for f in snap.iterdir():
        if f.is_symlink():
            blob = f.resolve()
            f.unlink()
            blob.rename(f)
    import onnx_asr
    onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")
    onnx_asr.load_vad("silero")
    print("prefetched")


if __name__ == "__main__":
    main()
