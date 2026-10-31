# Neural audio codecs (late 2026): benchmarks, evaluation metrics, and the open-weight landscape

Researcher notes, compiled 2026-10-30. Scope: evaluation methodology and the practical open-weight codecs. Out of scope (covered by other researchers): quality-vs-bitrate frontier analysis, architecture taxonomy, music codecs, and production standards.

Convention: "[spec per paper/card; not re-verified]" marks a number taken from the cited primary source that I could not re-check line by line in this session. Derived numbers (for example bitrate = codebooks × log2(codebook size) × frame rate) say so.

---

## Q1. Benchmarks and challenges: what they found, and who won

### Takeaway
No single leaderboard is authoritative. Codec-SUPERB, DASB, the "Discrete Audio Tokens" survey benchmark, CodecBench, and VERSA/ESPnet-Codec each measure different things: signal fidelity, downstream task utility, or token usefulness for LMs. The one large human-rated head-to-head is the 2025 Cisco-run LRAC challenge (186,720 crowdsourced ratings). It was won by ByteDance's IRIS codec (Track 1, transparency) and NJU's PR-Vocodec (Track 2, coding plus enhancement). It also found that high-resource open codecs still clearly beat compute-capped codecs at 1 kbps, and that objective metrics can be "deceived."

### Cited Findings

**2025 LRAC Challenge (Low-Resource Audio Codec; Cisco with CMU/UIUC/HUJI; ICASSP 2026 satellite workshop, May 4 2026)**
- Constraints. Both tracks have two bitrate modes: ultralow ≤1 kbps and low ≤6 kbps. Track 1 (transparency): receive-side compute ≤300 MFLOP/s, total ≤700 MFLOP/s, total latency ≤30 ms. Track 2 (enhancement codec): receive-side ≤600 MFLOP/s, total ≤2600 MFLOP/s, latency ≤50 ms. The test phase opened Oct 1 2025 and results were announced Oct 14 2025. — [LRAC 2025 Challenge Description, arXiv:2510.23312](https://arxiv.org/abs/2510.23312)
- Scale. 65 listening tests and 17 codecs at two bitrate modes, about 186,720 ratings, 24,140 crowd participants (not all unique). Methods were MUSHRA, MOS/DMOS (DCR), and DRT (Diagnostic Rhyme Test) for intelligibility. — [LRAC 2025 results page](https://crowdsourcing.cisco.com/lrac-challenge/2025/results); [LRAC 2026 workshop overview slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- Track 1 winners: 1st teamwzqaq (ByteDance, "IRIS codec"), 2nd nano_codec (Institute of Acoustics CAS, ByteDance, UCAS; "NanoCodec"), 3rd aitd_go (Anker; "PhoenixCodec"). Track 2 winners: 1st nju-aalab ("PR-Vocodec", progressive refinement training), 2nd xuyang (NJU; VQ-GAN with knowledge distillation), 3rd nanocodec ("Enhance-NanoCodec"). — [LRAC 2025 results](https://crowdsourcing.cisco.com/lrac-challenge/2025/results)
  - **Naming collision.** The LRAC "NanoCodec" (CAS/ByteDance) is not NVIDIA's NeMo NanoCodec.
  - PhoenixCodec paper: [arXiv:2510.21196](https://arxiv.org/pdf/2510.21196). ByteDance's IRIS poster self-reports 1st in Track 1 and 5th in Track 2: [IRIS poster](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-AS-18745-Poster-Wu-Iris-Low-Complexity-High-Efficiency-Neural-Network-Codec-For-Real-Time-Audio-Transmission.pdf).
- Track 1 had 6 submissions plus the baseline. The top 3 were within 2.7 MUSHRA points of each other, and all external entries beat the baseline by 8–30 points. In Track 2 (9 submissions plus baseline), nju-aalab won by about 5 points. — [LRAC overview slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- Comparison against high-resource baselines:
  - SOTA high-resource codecs cost one to two orders of magnitude more compute than the Track 1 codecs.
  - At 1 kbps (ULBR), high-resource codecs "significantly outperform" on both clean and mildly noisy/reverberant MUSHRA. At 6 kbps (LBR), the low-resource codecs are "comparable."
  - DualCodec relies on an offline w2v-BERT 2.0 front-end and has about 4 s latency without it.
  - There is a "transparency gap" for low-resource codecs in both quality and intelligibility. — [LRAC overview slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- Design patterns among top systems: STFT/mel front-ends; ConvNeXt, Conv2Former, or LKCAB blocks plus GRU/LSTM; Snake activations; GAN plus multi-stage training. Nearly all used factorized RVQ. There were no transformer-only or autoregressive decoders and no diffusion or flow-matching decoders. Differentiation came from data, architecture, and training rather than quantization. — [LRAC overview slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- Data findings:
  - nju-aalab trained on clean speech only. It had the best clean MUSHRA in Track 1 but fell to 4th–5th under noise/reverb.
  - Missing overlapping-speaker augmentation cost about 1.5 DMOS points.
  - Enhancement-capable codecs degraded intelligibility on average, even on clean input. Sustained consonants suffered most, with the largest clean-to-noisy drop at -87 DRT points (e.g., /S/ "shoes" vs /tS/ "choose"). — [LRAC overview slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- Evaluation lessons:
  - MUSHRA-1S (a single-stimulus MUSHRA variant) scaled to 17+ conditions.
  - DCR caveat: some outputs scored higher than their inputs, so listeners rated absolute quality rather than degradation.
  - The reference-based SCOREQ was the best objective metric (Pearson 0.87), but non-intrusive metrics saturate at high quality and PESQ "can be unreliable for generative neural codecs."
  - "Objective metrics remain advisory."
  - A second edition is planned to add 16/24 kbps and to mandate open-source implementations. — [LRAC overview slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)

**Codec-SUPERB (ACL 2024 Findings) and Codec-SUPERB @ SLT 2024 challenge**
- The original benchmark assesses codecs from both the application side (downstream tasks run on resynthesized audio) and the signal side. — [Codec-SUPERB, arXiv:2402.13071](https://arxiv.org/pdf/2402.13071)
- The SLT 2024 edition is training-free and uses license-free, subsampled datasets, with an open dev set and a hidden test set.
  - Application metrics: Whisper WER on LibriSpeech, ECAPA-TDNN EER on VoxCeleb1-O, emotion2vec accuracy on RAVDESS, CLAP accuracy on ESC-50.
  - Signal metrics: PESQ, STOI, SDR, mel loss.
  - Five participants in 13 settings: FunCodec, SemantiCodec (B1–B6), APCodec, AFACodec, SpeechTokenizer, with EnCodec as reference. — [Codec-SUPERB @ SLT 2024, arXiv:2409.14085](https://arxiv.org/html/2409.14085v1)
- Results:
  - AFACodec D3 at 7.5 kbps was strongest at mid bitrate: open-set WER 3.07% (raw audio 2.89%), EER 1.49% (raw 0.96%).
  - SemantiCodec kept audio-event information best at ≤2 kbps (hidden set B6 at 1.40 kbps: 70.37%).
  - At 4 kbps, APCodec beat SpeechTokenizer on almost all metrics, including WER.
  - No submission beat EnCodec on audio-event classification at 4–8 kbps. — [arXiv:2409.14085](https://arxiv.org/html/2409.14085v1)
- Metric correlations:
  - STOI correlated best with downstream utility: -0.807 with ASR WER, -0.891 with ASV EER, 0.920 with emotion accuracy.
  - Mel loss correlated weakly (0.254 with EER).
  - SemantiCodec's diffusion decoder causes time shifts that break signal metrics. — [arXiv:2409.14085](https://arxiv.org/html/2409.14085v1)

**DASB (Discrete Audio and Speech Benchmark; Mousavi et al., arXiv 2406.14294, published TMLR 2026)**
- Tasks: ASR, speaker ID/verification, emotion, keyword spotting, intent, enhancement, separation, and TTS. Tokenizers tested: semantic (discrete HuBERT/WavLM/wav2vec2), compression (EnCodec, DAC), and hybrid (SpeechTokenizer). — [DASB arXiv:2406.14294](https://arxiv.org/pdf/2406.14294); [TMLR 2026 listing](https://mlanthology.org/tmlr/2026/mousavi2026tmlr-dasb/)
- Findings:
  - Semantic tokens beat compression tokens on average across most discriminative and generative tasks.
  - Discrete tokens still trail continuous SSL features "substantially."
  - Discrete representations are less robust and need careful tuning. — [DASB](https://arxiv.org/html/2406.14294v1)

**"Discrete Audio Tokens: More Than a Survey!" (Mousavi et al., arXiv 2506.10274; ICLR 2026 poster)**
- Unified benchmark across speech, music, and general audio, covering reconstruction (via Codec-SUPERB and VERSA), downstream tasks, and acoustic LMs, with controlled ablations. — [arXiv:2506.10274](https://arxiv.org/html/2506.10274v3); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10014057)
- Secondhand summary, not checked against the primary tables:
  - Semantic distillation (SpeechTokenizer, Mimi) keeps WER low at low bitrate even as signal metrics fall.
  - SDR/SI-SNR are unreliable for low-bitrate neural codecs.
  - EnCodec was the strongest music/audio reconstructor.
  - Discrete WavLM led discriminative speech tasks. — [Moonlight review](https://www.themoonlight.io/review/discrete-audio-tokens-more-than-a-survey)

**Other 2025–2026 independent benchmarks**
- CodecBench (arXiv 2508.20660) is a comprehensive acoustic plus semantic codec benchmark. I could not retrieve its findings. — [CodecBench](https://arxiv.org/pdf/2508.20660)
- SITool (Interspeech 2025) benchmarks intelligibility (STOI, ESTOI, WER) for Codec2, AMR/AMR-WB, EVS, Lyra V2, DAC, SNAC, Mimi, and SemantiCodec. Search-snippet summary: Mimi at 1.1 kbps and DAC at 1.5 kbps score well. — [SITool, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/leschanowsky25_interspeech.pdf)
- African-speech robustness (Okocha & Grant, arXiv 2610.00154, Oct 2026) tested 7 codecs (DAC, EnCodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer) on AfriSpeech.
  - Dialog WER (Whisper-large-v3; uncompressed 17.3): DAC 16 kbps 28.26; LanguageCodec about 29–30; EnCodec 3 kbps 37.27; FocalCodec 12.5 Hz 57.02; SemantiCodec 0.31 kbps 76.22.
  - Speaker verification with ECAPA: DAC and LanguageCodec reach 0% EER. UniCodec 6.6 kbps is worst at 10.20%.
  - LoRA-adapting the DAC encoder (about 1% of parameters) cut WER from 21.21 to 17.81. — [arXiv:2610.00154](https://arxiv.org/html/2610.00154)

**Vendor-run tables, useful but not independent (each authors' own reproduction)**
- Qwen3-TTS tech report (arXiv 2601.15621), LibriSpeech test-clean:

  | Model | PESQ-WB | PESQ-NB | STOI | UTMOS | SIM |
  |---|---|---|---|---|---|
  | Qwen-TTS-Tokenizer-12Hz | 3.21 | 3.68 | 0.96 | 4.16 | 0.95 |
  | Mimi (16 codebooks) | 2.88 | 3.42 | 0.94 | 3.87 | 0.87 |
  | FireRedTTS-2 | 2.73 | 3.28 | 0.94 | 3.88 | 0.87 |
  | X-codec | 2.68 | 3.27 | 0.86 | 4.11 | 0.84 |
  | SpeechTokenizer | 2.60 | 3.05 | 0.92 | 3.90 | 0.85 |
  | X-codec 2 | 2.43 | 3.04 | 0.92 | 4.13 | 0.82 |
  | XY-Tokenizer | 2.41 | 3.00 | 0.91 | 3.98 | 0.83 |

  — [Qwen3-TTS report](https://arxiv.org/html/2601.15621)
- Kanade paper (arXiv 2602.00594), LibriSpeech test-clean, Parakeet WER / UTMOS / WavLM SIM. Ground truth is 1.9 / 4.07.

  | Model | Token rate (Hz) | WER (%) | UTMOS | SIM |
  |---|---|---|---|---|
  | DualCodec | 100 | 2.1 | 4.12 | 0.98 |
  | SpeechTokenizer | 400 | 2.1 | 3.90 | 0.98 |
  | PAST | 400 | 2.1 | 4.18 | 0.99 |
  | Kanade | 25 | 2.4 | 4.16 | 0.97 |
  | X-Codec 2 | 50 | 2.5 | 4.13 | 0.98 |
  | BiCodec | 50 | 2.5 | 4.18 | 0.98 |
  | Kanade | 12.5 | 3.3 | 4.17 | 0.97 |
  | StableCodec | 25 | 5.7 | 4.31 | 0.93 |
  | WavTokenizer | 40 | 9.4 | 3.57 | 0.92 |

  — [Kanade, arXiv:2602.00594](https://arxiv.org/html/2602.00594)

### Inferences
- The LRAC results are the strongest human-rated evidence. They support three conclusions:
  1. At about 6 kbps, speech coding is close to solved, even under 700 MFLOP/s.
  2. At ≤1 kbps, large open codecs (DualCodec, BigCodec-class models) still have a real perceptual edge.
  3. Robustness to noise and overlapping speech decides the ranking more than clean-speech scores do.
- Codec-SUPERB-style "resynthesize, then run downstream models" evaluation and DASB-style "train downstream models on tokens" evaluation answer different questions: transmission fidelity versus representation usefulness. A codec can win one and lose the other. Semantic tokenizers win DASB; acoustic codecs win signal fidelity.
- Vendor tables disagree with each other because each lab re-runs baselines with its own settings: codebook count used for Mimi, resampling, ASR model. In Qwen's table X-Codec2 has the lowest SIM (0.82); in Kanade's it scores 0.98, because the SIM models differ. Absolute numbers should not be compared across papers.

### Gaps
- **ERSB.** I found no benchmark by this name. It may be a misremembered acronym, or something too new or obscure to index.
- No Codec-SUPERB edition after SLT 2024 (Interspeech 2025/2026) surfaced. Whether the online Codec-SUPERB leaderboard is still maintained in 2026 is unverified.
- I could not retrieve full LRAC per-team MUSHRA scores or bitrates (they are on Codabench). Nor could I tell whether winning LRAC systems released weights; the second edition plans to *mandate* open source, which implies the first did not.
- I did not obtain CodecBench results or the primary tables of the "More Than a Survey" benchmark.
- Interspeech 2026 and ICASSP 2026 codec challenges other than LRAC: none found.

---

## Q2. Metrics: what is used, failure modes, and correlation with human judgments

### Takeaway
For codec speech, the best-supported objective proxies of MUSHRA are reference-based SCOREQ (Pearson about 0.80–0.94 across three studies) and, surprisingly, PESQ on clean speech (0.69–0.90). No-reference MOS predictors (NISQA, DNSMOS, and to a lesser degree UTMOS) correlate poorly. They saturate at high quality, and they can rate codec output above the reference, which rewards "clean-sounding but wrong" generative decoders. Downstream fidelity metrics (WER/CER, speaker SIM/EER) are essential, but the ranking depends on which ASR or ASV model you choose.

### Cited Findings

**Correlation with listening tests**
- Lanzendörfer & Grötschla (arXiv 2511.19734) ran a crowdsourced MUSHRA test on Mabyduck with 11 listeners for speech-only and 17 for mixed audio. Codecs: DAC, EnCodec 6 kbps, Multi-Band Diffusion, Mimi (4.4 kbps), SNAC (44.1 kHz, 2.6 kbps), Vocos.

  | Metric | Pearson, speech-only | Pearson, speech+background |
  |---|---|---|
  | SCOREQ | 0.937 | 0.869 |
  | PESQ | 0.886 | 0.903 |
  | ViSQOL-Speech | 0.850 | 0.593 |
  | WARP-Q | 0.847 | 0.632 |
  | ViSQOL-Audio | 0.691 | 0.644 |
  | DNSMOS | 0.683 | -0.073 |
  | NISQA | 0.563 | 0.193 |

  The authors recommend PESQ plus SCOREQ. Experienced listeners sometimes could not tell DAC from the reference. — [arXiv:2511.19734](https://arxiv.org/abs/2511.19734)
- Lechler et al. (Interspeech 2025, Cisco) compared crowdsourced MUSHRA on Prolific/MTurk against internal tests. Codecs: Webex AI Codec 6 kbps, EnCodec 6 kbps, Opus 16 kbps, EVS 6 kbps, with an Opus 6 kbps anchor.

  | Metric | Overall | DSP codecs | DNN codecs |
  |---|---|---|---|
  | PESQ | 0.69 | 0.79 | 0.58 |
  | POLQA | 0.72 | 0.78 | 0.69 |
  | ViSQOL | 0.55 | 0.69 | 0.30 |
  | NISQA | 0.18 | 0.37 | 0.18 |
  | DNSMOS-SIG | 0.21 | 0.28 | 0.29 |
  | SCOREQ (distance, so negative = good) | -0.80 | -0.79 | -0.79 |

  Conclusion: PESQ and POLQA *underestimate DNN codecs*. SCOREQ is consistent across architectures. Crowdsourced MUSHRA with screening reproduces internal-lab rankings. — [Lechler et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/lechler25_interspeech.pdf)
- Mack et al. (ICASSP 2026, arXiv:2509.24457) assessed 45 metrics on clean speech. Neural metrics (SCOREQ, UTMOS) had the highest Pearson correlation; they recommend SCOREQ, UTMOS, and "ce." A secondary summary gives SCOREQ about 0.87 and PESQ/WARP-Q about 0.73 (unverified). — [ResearchGate entry](https://www.researchgate.net/publication/395969272_Assessing_speech_quality_metrics_for_evaluation_of_neural_audio_codecs_under_clean_speech_conditions); the LRAC slides independently state scoreq_ref best at Pearson 0.87 ([slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf))
- A 2020 vocoder study found generative coders correlate poorly between PESQ and MUSHRA while parametric coders correlate well. This is the origin of "PESQ penalizes generative decoders." — [arXiv:2003.11882](https://arxiv.org/pdf/2003.11882)
- MUSHRA-1S (Lechler & Balić, ICASSP 2026, arXiv:2509.19219) is a scalable single-stimulus MUSHRA for top-tier systems, used in LRAC. — [LRAC slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)

**Failure modes documented with numbers**
- *No-reference MOS above the reference ("hallucination reward"):*
  - In Kanade's table, StableCodec has the highest UTMOS (4.31, above ground truth 4.07) but WER 5.7% against a 1.9% reference. — [arXiv:2602.00594](https://arxiv.org/html/2602.00594)
  - NISQA scored codec output above the uncompressed reference on AfriNames (3.25 vs 2.92). WavTokenizer S-600 ranked 1st of 28 on NISQA but was among the worst on WER (68.46). FocalCodec 12.5 Hz ranked 1st of 31 on UTMOS yet did poorly on EER. — [arXiv:2610.00154](https://arxiv.org/html/2610.00154)
- *Weak downstream correlation of MOS predictors.* Spearman ρ with dialog WER: ViSQOL -0.74, STOI -0.72, NISQA -0.43, UTMOS -0.26. F0-RMSE was 0.85. — [arXiv:2610.00154](https://arxiv.org/html/2610.00154)
- *Spectrograms can look clean while speech is unintelligible.* LRAC noted objective quality metrics "may be deceived," and that non-intrusive metrics saturate at high quality. — [LRAC slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- *Signal metrics break under time shifts.* SDR and mel loss correlate weakly with utility, and SemantiCodec's diffusion decoder causes misalignment. — [Codec-SUPERB @ SLT 2024](https://arxiv.org/html/2409.14085v1). SDR/SI-SNR are unreliable for low-bitrate codecs (secondhand). — [Moonlight review of 2506.10274](https://www.themoonlight.io/review/discrete-audio-tokens-more-than-a-survey)
- *Backend dependence of speaker metrics.* WavTokenizer S-600 EER is 16.1% under ECAPA but 2.4% under MFCC-GMM. FocalCodec moves from 4–11% to 14–27%, so the ranking flips with the backend. — [arXiv:2610.00154](https://arxiv.org/html/2610.00154)
- *Same metric, different values across papers.* X-Codec2 SIM is 0.82 in the Qwen3-TTS table and 0.98 in Kanade's. — [Qwen3-TTS](https://arxiv.org/html/2601.15621); [Kanade](https://arxiv.org/html/2602.00594)
- *DCR/MOS listening tests.* Listeners rated some outputs above the input, judging absolute quality rather than degradation. — [LRAC slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)
- *Enhancement codecs.* They can improve MOS-type scores in noise while hurting intelligibility, especially on sustained consonants per DRT. — [LRAC slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)

**Metric usage conventions seen in current papers**
- Speech-tokenizer papers typically report PESQ-WB/NB, STOI, UTMOS, and speaker SIM (WavLM-SV cosine) on LibriSpeech test-clean, plus WER from Whisper or Parakeet. — [Qwen3-TTS](https://arxiv.org/html/2601.15621); [Kanade](https://arxiv.org/html/2602.00594)
- NVIDIA reports Squim MOS, PESQ, mel distance, SECS, and CER on MLS and DAPS. — [NanoCodec card](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
- Challenge-grade evaluation uses MUSHRA, DCR/DMOS, and DRT. — [LRAC results](https://crowdsourcing.cisco.com/lrac-challenge/2025/results)

### Inferences
- A defensible minimal protocol for speech codecs:
  - **Quality:** SCOREQ (reference mode) plus PESQ-WB as the primary proxies.
  - **Content:** WER/CER from two different ASR models (for example Whisper-large-v3 and Parakeet).
  - **Speaker:** SIM from two backends (WavLM-SV or ECAPA).
  - **Prosody:** F0-RMSE or F0-corr.
  - **Reporting:** UTMOS/NISQA/DNSMOS only as secondary "naturalness" indicators, flagging any score above the reference.
  - **Confirmation:** a small MUSHRA (or MUSHRA-1S) test.
- PESQ's reputation as bad for generative codecs is half true. PESQ underestimates DNN codecs in absolute terms (Lechler: DNN subset Pearson 0.58). Yet on homogeneous sets of modern GAN codecs it ranks systems well (Lanzendörfer: 0.89–0.90). It is likely most dangerous for diffusion or flow decoders and codecs with phase or time drift. That last point is my inference: MBD had the lowest PESQ of the six codecs and was also rated lowest in MUSHRA.
- All of these correlation studies use small numbers of systems (4–6 codecs). Their Pearson values carry large uncertainty, and none reports intervals. Treat the rank order "SCOREQ ≈ PESQ > ViSQOL-Speech > NISQA/DNSMOS" as robust, and the exact coefficients as not.

### Gaps
- No study found that measured UTMOS (or UTMOSv2) correlation with MUSHRA specifically on generative and diffusion codecs. The Mack et al. result that favors UTMOS is clean-speech and non-generative-specific, and I could not verify its numbers.
- STOI, SI-SDR, and FAD correlations with human ratings for codecs: no dedicated 2025–2026 study found. FAD is more relevant to the music researchers.
- POLQA is licensed and closed, so most academic studies omit it. Lechler et al. (POLQA 0.72) is the main codec-specific data point.

---

## Q3. Open-weight codecs available today: specs, licenses, and standing

### Takeaway
For speech-LM tokenization, the practical frontier in late 2026 is:
- **Low-frame-rate causal multi-codebook models:** Qwen3-TTS-Tokenizer-12Hz (Apache-2.0, best vendor-reported reconstruction), Mimi (CC-BY-4.0, by far the most adopted), and NVIDIA NanoCodec (NVIDIA Open Model License).
- **Single-codebook 50 Hz models:** XCodec2 (CC-BY-NC), NeuCodec (Apache-2.0), and BiCodec (CC-BY-NC-SA).
- **12.5–25 Hz single-token models:** Kanade (MIT).

For pure low-bitrate compression with permissive licenses, the options are DAC (MIT) and SNAC (MIT), with DualCodec/BigCodec-class models leading at ≤1 kbps in human tests but lacking streaming. Licenses vary widely and matter: NC licenses apply to XCodec2 and BiCodec, and Stable Codec uses a custom gated license.

### Cited Findings: model table
Downloads are Hugging Face all-time-ish counts as returned by the Hub API on 2026-10-30.

| Model (HF repo) | License | SR (kHz) | Frame rate / bitrate | Quantizer | Streaming/causal | Params | HF downloads | Notes / independent standing |
|---|---|---|---|---|---|---|---|---|
| **Mimi** ([kyutai/mimi](https://hf.co/kyutai/mimi)) | CC-BY-4.0 | 24 | 12.5 Hz; 1.1 kbps at 8 codebooks, up to 32 codebooks [spec per [Moshi paper](https://arxiv.org/abs/2410.00037); not re-verified]. Run at 4.4 kbps in Lanzendörfer, 16 codebooks (about 2.2 kbps) in the Qwen table | split RVQ with semantic (WavLM-distilled) first codebook | causal, streaming | 96.2M | 22.6M | Most-used speech-LM codec (Moshi, Sesame CSM, Dia2 spaces). Good intelligibility at 1.1 kbps in SITool ([ISCA](https://www.isca-archive.org/interspeech_2025/leschanowsky25_interspeech.pdf)). Qwen table: PESQ-WB 2.88, SIM 0.87 ([2601.15621](https://arxiv.org/html/2601.15621)) |
| **Qwen3-TTS-Tokenizer-12Hz** ([Qwen/Qwen3-TTS-Tokenizer-12Hz](https://hf.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)) | Apache-2.0 | not stated on card | 12.5 Hz, 16 × 2048 codebooks, about 2.2 kbps (derived) | multi-codebook | fully causal, no look-ahead; 4-token (320 ms) packets, about 4 ms decode per packet | 170.6M (card says 0.2B) | 1.2M (released Jan 2026) | Vendor table: PESQ-WB 3.21, STOI 0.96, UTMOS 4.16, SIM 0.95, best in its table ([report](https://arxiv.org/html/2601.15621)). The sister 25 Hz single-codebook (32,768) tokenizer needs DiT plus BigVGAN look-ahead |
| **NVIDIA NanoCodec** ([nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps](https://hf.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps); also 1.78 kbps-12.5 fps and 0.6 kbps-12.5 fps) | NVIDIA Open Model License (commercial OK) | 22.05 | 21.5 fps / 1.89 kbps (8 codebooks); 12.5 fps / 1.78 kbps (13 codebooks); 12.5 fps / 0.6 kbps (4 codebooks, speaker-limited fine-tuning only) | FSQ [8,7,6,6] | non-causal encoder, causal decoder | 62M (card); GGUF section says 39.3M | 114.8K | Used by Magpie-TTS. Card numbers: MLS PESQ 2.837, CER 2.43; DAPS PESQ 3.156 ([card](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)); paper [arXiv:2508.05835](https://arxiv.org/abs/2508.05835). The older [nvidia/audio-codec-22khz](https://hf.co/nvidia/audio-codec-22khz) and [audio-codec-44khz](https://hf.co/nvidia/audio-codec-44khz) are low-frame-rate audio codecs, license "other" |
| **XCodec2** ([HKUSTAudio/xcodec2](https://hf.co/HKUSTAudio/xcodec2)) | **CC-BY-NC-4.0** | 16 | 50 Hz, single 65,536-entry codebook, 0.8 kbps | single FSQ | not streaming (per paper; not re-verified) | 822.7M | 530.3K | Backbone of Llasa TTS. Kanade table: WER 2.5, SIM 0.98. Qwen table: PESQ-WB 2.43, SIM 0.82 |
| **NeuCodec** ([neuphonic/neucodec](https://hf.co/neuphonic/neucodec)) | Apache-2.0 (gated) | 16 in, 24 out | 50 Hz, 0.8 kbps (16 bits/token) | single FSQ | not stated | 629.9M (0.6B) | 1.8M | XCodec2-style codec used by NeuTTS ([card](https://huggingface.co/neuphonic/neucodec); paper [arXiv:2509.09550](https://arxiv.org/abs/2509.09550)). No independent benchmark found |
| **BiCodec** (inside [SparkAudio/Spark-TTS-0.5B](https://hf.co/SparkAudio/Spark-TTS-0.5B)) | **CC-BY-NC-SA-4.0** | 16 | 50 Hz semantic tokens plus fixed global speaker tokens (about 0.65 kbps per [paper](https://arxiv.org/abs/2503.01710); not re-verified) | single-codebook semantic plus global | no | (in 0.5B TTS) | 64.1K (TTS repo) | Kanade table: WER 2.5, UTMOS 4.18, SIM 0.98 |
| **Kanade** ([frothywater/kanade-25hz](https://hf.co/frothywater/kanade-25hz); also a 12.5 Hz variant) | MIT (HF tag) | 24 (LibriTTS training; Vocos decoder) | 12.5 or 25 Hz, one token per step | single-layer FSQ | not stated | about 120M trainable, about 210M total (frozen WavLM Base+ plus Vocos); HF says 117.7M | 5.1K | Released Feb 2026 ([arXiv:2602.00594](https://arxiv.org/abs/2602.00594)). 25 Hz: WER 2.4, UTMOS 4.16, SIM 0.97 (own table) |
| **DualCodec** ([amphion/dualcodec](https://hf.co/amphion/dualcodec)) | license not shown on HF card | 24 | 12.5/25 Hz variants (per [paper](https://arxiv.org/abs/2505.13000); not re-verified); 100 Hz total token rate in Kanade table | semantic (w2v-BERT 2.0) first layer plus RVQ | **no**: offline w2v-BERT front-end, about 4 s latency without it ([LRAC slides](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)) | not shown | not shown | Used as a high-resource reference in LRAC (beat compute-capped codecs at 1 kbps). Kanade table: WER 2.1, SIM 0.98 |
| **Higgs Audio v2 tokenizer** ([bosonai/higgs-audio-v2-tokenizer](https://hf.co/bosonai/higgs-audio-v2-tokenizer)) | "other" (Boson license) | not verified | not verified | not verified | not verified | 201.4M | 420.0K | Updated Oct 2026; transformers-native. No independent benchmark found |
| **MiMo-Audio-Tokenizer** ([XiaomiMiMo/MiMo-Audio-Tokenizer](https://hf.co/XiaomiMiMo/MiMo-Audio-Tokenizer)) | MIT | not verified | not verified | not verified | not verified | 1,295.8M | 41.4K | Updated 2026-10-28 (very recent). No independent benchmark found |
| **SNAC** ([hubertsiuzdak/snac_24khz](https://hf.co/hubertsiuzdak/snac_24khz); 32/44 kHz variants) | MIT | 24 (speech); 32/44.1 (music) | multi-scale RVQ (12/23/47 Hz for 24 kHz, 0.98 kbps per [paper](https://arxiv.org/abs/2410.14411); not re-verified); 44.1 kHz at 2.6 kbps used in Lanzendörfer | multi-scale RVQ | not streaming by design (not re-verified) | not shown | 16.1M | Orpheus TTS and Maya1 backbone. Lanzendörfer MUSHRA: below DAC, Mimi, and Vocos on speech |
| **WavTokenizer** ([novateur/WavTokenizer-large-speech-75token](https://hf.co/novateur/WavTokenizer-large-speech-75token)) | MIT | 24 | 40 or 75 Hz single codebook (4096; about 0.48/0.9 kbps per [paper](https://arxiv.org/abs/2408.16532); not re-verified) | single VQ | no | not shown | not shown | **Poor independent results:** Kanade WER 9.4%, SIM 0.92; African-speech S-600 WER 68.46 despite top NISQA |
| **Stable Codec** ([stabilityai/stable-codec-speech-16k](https://hf.co/stabilityai/stable-codec-speech-16k)) | "other" (Stability community license), **gated** | 16 | 25 Hz | residual FSQ | not verified | 953.4M | 7.7K | Highest UTMOS (4.31) but WER 5.7% in the Kanade table, the textbook hallucination case ([Kanade](https://arxiv.org/html/2602.00594)) |
| **DAC** ([descript/descript-audio-codec](https://hf.co/descript/descript-audio-codec); MLX ports exist) | MIT | 16 / 24 / 44.1 | 44.1 kHz model: 86 Hz, up to 9 codebooks, 7.74 kbps ([Lanzendörfer](https://arxiv.org/abs/2511.19734)) | RVQ (factorized) | no | not shown | (via mirrors) | Best MUSHRA in Lanzendörfer (near-transparent on speech). Best ASR/ASV preservation in the African-speech study. Good at 1.5 kbps in SITool |
| **EnCodec** ([facebook/encodec_24khz](https://hf.co/facebook/encodec_24khz)) | license not tagged on card (EnCodec repo historically CC-BY-NC for weights; not verified) | 24 / 48 | 75 Hz, 1.5–24 kbps | RVQ | causal 24 kHz model | 23.3M | 36.5M | Degrades at ≤2 kbps (Codec-SUPERB). Still a strong general-audio and music reconstructor (secondhand) |
| **SpeechTokenizer** ([OpenMOSS-Team/SpeechTokenizer](https://hf.co/OpenMOSS-Team/SpeechTokenizer)) | not tagged | 16 | 50 Hz × 8 RVQ (about 4 kbps, derived) | HuBERT-distilled RVQ | no | not shown | not shown | Superseded at 4 kbps by APCodec in Codec-SUPERB SLT. Qwen table PESQ-WB 2.60 |
| **LongCat-Audio-Codec** ([meituan-longcat/LongCat-Audio-Codec](https://hf.co/meituan-longcat/LongCat-Audio-Codec)) | MIT | not verified | not verified | not verified | not verified | not verified | not verified | Oct 2025 release ([arXiv:2510.15227](https://arxiv.org/abs/2510.15227)) |
| **CosyVoice2 speech tokenizer** (inside [FunAudioLLM/CosyVoice2-0.5B](https://hf.co/FunAudioLLM/CosyVoice2-0.5B)) | Apache-2.0 | 16 (not verified) | 25 Hz FSQ semantic tokens (not verified) | FSQ supervised by ASR | token-to-wave via flow matching, chunk-streaming (not verified) | — | 45.7K | ASR-supervised semantic tokenizer, not a waveform codec; needs a flow-matching decoder |

**Production-ish open codecs:** Lyra V2 appears among SITool's evaluated codecs ([ISCA](https://www.isca-archive.org/interspeech_2025/leschanowsky25_interspeech.pdf)). The LRAC description cites Opus and AMR-WB as the classical low-compute, low-latency reference whose quality degrades in noise ([arXiv:2510.23312](https://arxiv.org/abs/2510.23312)). I did not research Opus 1.5 DRED, LACE/NoLACE, LPCNet, or FARGAN in this session (see Gaps).

### Recommendations by use case (inference from the cited findings)
- **(i) Pure speech compression at low bitrate:**
  - *Permissive and compute-light:* DAC 16/24 kHz at 1.5–6 kbps (MIT; strongest independent ASR/ASV preservation and MUSHRA), or Mimi at 1.1 kbps (CC-BY, causal and streaming, good intelligibility).
  - *At ≤1 kbps, offline, compute no object:* DualCodec, BigCodec, or XCodec2/NeuCodec. LRAC confirmed that high-resource codecs beat constrained ones there. Check the NC licenses (XCodec2) and DualCodec's latency.
  - *Under real-time CPU or embedded budgets:* the LRAC systems are the reference. Weights availability is unverified.
- **(ii) Tokenization for speech LMs:**
  - *Low-frame-rate, causal, multi-codebook (for Moshi/CSM-style delay or depth transformers):* Qwen3-TTS-Tokenizer-12Hz (Apache-2.0) or Mimi.
  - *Single-codebook, for plain next-token LMs:* XCodec2 (NC), NeuCodec (Apache-2.0), BiCodec (NC-SA), Kanade (MIT; 12.5–25 Hz), or DualCodec (12.5 Hz).
  - DASB evidence says semantic-distilled tokens beat pure acoustic tokens for understanding tasks.
  - Avoid WavTokenizer for content-critical uses.
- **(iii) General audio:** DAC 44.1 kHz and EnCodec remain the safe open baselines. Codec-SUPERB found nothing beat EnCodec on audio events at 4–8 kbps, and SemantiCodec preserves events at ≤1.4 kbps. NVIDIA audio-codec-44khz is an alternative. Defer to the music researcher's notes.

### Gaps
- Exact licenses for DualCodec, SpeechTokenizer, EnCodec weights, and Higgs v2 (Boson "other" license terms) were not verified on cards.
- The BigCodec HF repo ID was not found under the ID I tried. Specs (16 kHz, 80 Hz, single 8192 codebook, about 1.04 kbps, about 159M params) are from memory of [arXiv:2409.05377](https://arxiv.org/abs/2409.05377) and not verified this session.
- FocalCodec weights: `lucadellalib/focalcodec` was not found on HF. The project is at [github.com/lucadellalib/focalcodec](https://github.com/lucadellalib/focalcodec) (not checked). Variants of 12.5/25/50 Hz and 0.16–0.65 kbps are confirmed by [arXiv:2610.00154](https://arxiv.org/html/2610.00154).
- MiMo-Audio-Tokenizer and Higgs v2 tokenizer specs (sample rate, frame rate, codebooks, streaming) were not retrieved.
- Opus 1.5 DRED, LACE/NoLACE, LPCNet/FARGAN, and Lyra v2 details and licenses were not researched for lack of budget. They are BSD/Apache in their upstream repos to my knowledge, but unverified here.
- "Mimi 2" or other 2026 Kyutai codec releases: `kyutai/mimi-2` does not exist on HF. The Mimi card was last updated Jul 2025.
- No independent (non-vendor) benchmark yet includes Qwen3-TTS-Tokenizer, NanoCodec, NeuCodec, MiMo-Audio-Tokenizer, or Higgs v2 alongside Mimi and XCodec2 under one protocol.

---

## Q4. Toolkits for easy comparison

### Takeaway
VERSA (CMU WAVLab, ACL 2025 demo) is the most comprehensive metric toolkit and is what recent benchmarks use. Codec-SUPERB provides a ready downstream-task protocol. ESPnet-Codec provides unified training and evaluation recipes. For quick listening comparisons there are HF Spaces that wrap multiple codecs.

### Cited Findings
- VERSA offers 65 metrics and 729 variants via a Python interface with configurable dependencies. It supports matching and non-matching references, text transcripts, and captions, and has demonstrated use in audio coding, TTS, enhancement, singing, and music generation. GitHub: [shinjiwlab/versa](https://github.com/shinjiwlab/versa) per the ACL Anthology; an alternate wavlab-speech path is reported. — [VERSA](https://awesomepapers.io/speech-audio/papers/shi2024versa)
- ESPnet-Codec: "Comprehensive Training and Evaluation of Neural Codecs for Audio, Music, and Speech" (2024). — [arXiv:2409.15897](https://arxiv.org/pdf/2409.15897)
- Codec-SUPERB provides training-free resynthesis-plus-downstream evaluation with an open and hidden split, and is used by the "More Than a Survey" benchmark alongside VERSA. — [arXiv:2409.14085](https://arxiv.org/html/2409.14085v1); [arXiv:2506.10274](https://arxiv.org/html/2506.10274v3)
- DASB code and leaderboards: [project site](https://poonehmousavi.github.io/DASB-website) (via [DASB paper](https://arxiv.org/pdf/2406.14294)).
- Transformers-native codecs ease swapping: Mimi, EnCodec, SNAC, and Higgs v2 tokenizer are tagged `transformers` / `AutoModel` on the Hub. A "compare-codec" Space ([twangodev/compare-codec](https://hf.co/spaces/twangodev/compare-codec)) and "audio-to-audio-models-on-the-hub" ([eustlb](https://hf.co/spaces/eustlb/audio-to-audio-models-on-the-hub)) link Mimi, XCodec2, NeuCodec, and Qwen3 tokenizers. — Hub metadata via the HF API (see model links above)
- LRAC organizers publish a FLOP/latency calculation spreadsheet and baselines in their repository, useful for compute-budgeted comparisons. — [arXiv:2510.23312](https://arxiv.org/abs/2510.23312)
- Crowdsourced MUSHRA tooling: Mabyduck was used by Lanzendörfer ([arXiv:2511.19734](https://arxiv.org/abs/2511.19734)). Prolific was validated as matching internal labs by Lechler et al. ([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/lechler25_interspeech.pdf)).

### Inferences
- The practical stack is VERSA for metrics (including SCOREQ, PESQ, UTMOS, speaker SIM, and WER), plus Codec-SUPERB-style resynthesis for downstream tasks, plus a Prolific MUSHRA-1S run for final claims. Pin the ASR and SV models and report them, because values shift substantially across backends.

### Gaps
- Whether `descript-audiotools` is still maintained in 2026, and the current ESPnet-Codec model zoo contents, were not checked.
- Whether SCOREQ, UTMOSv2, and Squim are all in current VERSA releases was not verified, though VERSA's 65-metric count makes it likely.
