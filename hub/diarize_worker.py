"""sherpa-onnx speaker diarization, run in a child process of mimi-hub.

OfflineSpeakerDiarization.process() holds the GIL for its whole run (~2 min per 45 min of audio). In the hub
process that starved the thread launching the Mimi encoder's GPU kernels: 17 s alone, 128 s alongside it.
"""

import numpy as np

_diar = None


def segments(npy_path, models_dir, threshold):
    """[(start_s, end_s, speaker)] for the 16 kHz mono float32 audio saved at [npy_path]."""
    global _diar
    import sherpa_onnx as so
    if _diar is None:
        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=f"{models_dir}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
                num_threads=8),
            embedding=so.SpeakerEmbeddingExtractorConfig(
                model=f"{models_dir}/wespeaker_en_voxceleb_resnet34_LM.onnx", num_threads=8),
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.5)
        if not cfg.validate():
            raise RuntimeError(f"diarization models missing under {models_dir}")
        _diar = so.OfflineSpeakerDiarization(cfg)
    wav = np.load(npy_path)
    return [(s.start, s.end, s.speaker) for s in _diar.process(wav).sort_by_start_time()]
