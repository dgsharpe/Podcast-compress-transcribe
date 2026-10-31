# Low and very-low bitrate neural SPEECH codecs: quality-vs-bitrate frontier (state as of 2026-10-30)

Scope: speech only, ~6 kbps down to sub-200 bps. General audio/music and production codecs (Opus, Lyra, 3GPP) are out of scope except for comparison. All bitrates below are **raw token bitrates** (frame rate x sum of log2(codebook size)) unless stated; essentially no paper in this space reports entropy-coded rates. "dWER" = WER of ASR on the reconstruction relative to ASR on the original (Whisper-based in the FocalCodec/ZipCodec family). "Sim" values are NOT comparable across papers (different speaker-embedding models / scales; see Q5).

Research note on method: ~18 tool calls. Primary sources are arXiv HTML tables fetched directly. Where a number came from a search snippet rather than a fetched table, it is marked "(snippet)".

---

## Q1. Which codecs lead at ~0.6–6 kbps, and what numbers do they report?

### Takeaway
At about 0.65–1.1 kbps, single-codebook 16 kHz codecs (**BigCodec** 1.04 kbps, **X-Codec2** 0.8 kbps, **BiCodec** 0.65 kbps, **FocalCodec@50** 0.65 kbps) and the low-frame-rate **DualCodec** (0.75–0.93 kbps) now reach UTMOS about 4.0–4.2, roughly the same as the clean LibriSpeech reference (about 4.09). They also keep intelligibility close to the reference (dWER about 2–3%). Older RVQ codecs at the same rate fall far behind: EnCodec at 1.5 kbps and DAC at 1 kbps score UTMOS 1.3–1.6. At 6 kbps, DAC with 12 codebooks reaches PESQ-WB 4.0 and SIM 0.98, so 6 kbps is effectively transparent for speech. The quality fight is now below about 1 kbps.

### Cited Findings

**Table A — FocalCodec paper (NeurIPS 2025; arXiv 2502.04465v2, rev. 24 Oct 2025), LibriSpeech test-clean, single test harness for all codecs** — [FocalCodec](https://arxiv.org/html/2502.04465)

| Codec | SR (kHz) | kbps | Codebooks | Frame rate (Hz) | UTMOS | dWER % | Sim |
|---|---|---|---|---|---|---|---|
| Reference | – | – | – | – | 4.09 | 0.00 | 100 |
| EnCodec | 24 | 1.50 | 2x1024 | 75 | 1.58 | 8.08 | 93.8 |
| DAC | 16 | 1.00 | 2x1024 | 50 | 1.29 | 20.04 | 89.2 |
| WavLM6-KM (semantic k-means) | 16 | 0.45 | 1x512 | 50 | 3.75 | 6.20 | 90.0 |
| SpeechTokenizer | 16 | 1.00 | 2x1024 | 50 | 2.28 | 5.14 | 91.6 |
| SemantiCodec | 16 | 0.65 | 2x8192 | 25 | 2.91 | 8.97 | 96.0 |
| Mimi (5 cb) | 24 | 0.69 | 5x2048 | 12.5 | 3.29 | 5.73 | 96.0 |
| WavTokenizer | 24 | 0.48 | 1x4096 | 40 | 3.78 | 11.55 | 95.4 |
| BigCodec | 16 | 1.04 | 1x8192 | 80 | 4.11 | 2.55 | 98.5 |
| Stable Codec (TAAE) | 16 | 0.70 | 2x15625 | 25 | 4.32 | 4.97 | 94.7 |
| FocalCodec@50 | 16 | 0.65 | 1x8192 (binary, 13-bit) | 50 | 4.05 | 2.18 | 97.4 |
| FocalCodec@25 | 16 | 0.33 | 1x8192 | 25 | 4.14 | 3.30 | 96.3 |
| FocalCodec@12.5 | 16 | 0.16 | 1x8192 | 12.5 | 4.22 | 7.94 | 93.9 |

- In the same paper, on the multilingual (MLS, 700 utterances) test (reference UTMOS 2.84), the dWER results are: FocalCodec@50 12.57, BigCodec 15.24, FocalCodec@25 19.78, Mimi 30.96, WavTokenizer 49.73, FocalCodec@12.5 54.15, and Stable Codec 56.99. Codecs trained mostly on English with heavy semantic content collapse on intelligibility in other languages. — [FocalCodec](https://arxiv.org/html/2502.04465)
- FocalCodec does not report PESQ or STOI, because it argues signal metrics correlate poorly with perceived quality. Its MOS test (33 valid raters, 10 LibriSpeech clips) appears only as a figure. In that test, FocalCodec@50 is "comparable to BigCodec and Stable Codec." — [FocalCodec](https://arxiv.org/html/2502.04465)

**Table B — Spark-TTS / BiCodec paper (arXiv 2503.01710v1, 3 Mar 2025), LibriSpeech test-clean** — [Spark-TTS](https://arxiv.org/html/2503.01710)

| Model | Codebook | Nq | Tokens/s | bps | STOI | PESQ-NB | PESQ-WB | UTMOS | SIM |
|---|---|---|---|---|---|---|---|---|---|
| EnCodec | 1024 | 8 | 600 | 6000 | 0.94 | 3.17 | 2.75 | 3.07 | 0.89 |
| DAC | 1024 | 12 | 600 | 6000 | 0.95 | 4.15 | 4.01 | 4.00 | 0.98 |
| EnCodec | 1024 | 2 | 150 | 1500 | 0.84 | 1.94 | 1.56 | 1.58 | 0.60 |
| Mimi | 2048 | 8 | 100 | 1100 | 0.91 | 2.80 | 2.25 | 3.56 | 0.73 |
| BigCodec | 8192 | 1 | 80 | 1040 | 0.94 | 3.27 | 2.68 | 4.11 | 0.84 |
| DAC | 1024 | 2 | 100 | 1000 | 0.73 | 1.40 | 1.14 | 1.29 | 0.32 |
| SpeechTokenizer | 1024 | 2 | 100 | 1000 | 0.77 | 1.59 | 1.25 | 2.28 | 0.36 |
| X-codec | 1024 | 2 | 100 | 1000 | 0.86 | 2.88 | 2.33 | 4.21 | 0.72 |
| WavTokenizer | 4096 | 1 | 75 | 900 | 0.89 | 2.64 | 2.14 | 3.94 | 0.67 |
| X-codec2 | 65536 | 1 | 50 | 800 | 0.92 | 3.04 | 2.43 | 4.13 | 0.82 |
| StableCodec | 15625 | 2 | 50 | 697 | 0.91 | 2.91 | 2.24 | 4.23 | 0.62 |
| Single-Codec | 8192 | 1 | 23.4 | 304 | 0.86 | 2.42 | 1.88 | 3.72 | 0.60 |
| BiCodec | 8192 | 1 | 50 | 650 | 0.92 | 3.13 | 2.51 | 4.18 | 0.80 |

**Table C — DualCodec (Interspeech 2025; arXiv 2505.13000v2, rev. 1 Oct 2025), LibriSpeech test-clean, includes MUSHRA (authors' own test)** — [DualCodec](https://arxiv.org/html/2505.13000)

| System | Params | kbps | Tok/s | #VQ | PESQ-NB | PESQ-WB | STOI | UTMOS | MUSHRA |
|---|---|---|---|---|---|---|---|---|---|
| DAC-official 75 Hz | 74M | 0.75 | 75 | 1 | 1.46 | 1.18 | 0.75 | 1.32 | 26.0 |
| EnCodec 75 Hz | 15M | 1.5 | 150 | 2 | 1.92 | 1.54 | 0.84 | 1.55 | 36.2 |
| SpeechTokenizer 50 Hz | 103M | 1.0 | 100 | 2 | 1.42 | 1.15 | 0.70 | 1.81 | 35.9 |
| WavTokenizer-large 75 Hz | 71M | 0.90 | 75 | 1 | 2.54 | 2.05 | 0.89 | 3.87 | 81.0 |
| Mimi 12.5 Hz | 78M | 0.83 | 75 | 6 | 2.51 | 1.99 | 0.89 | 3.43 | 72.8 |
| DualCodec 25 Hz (16384+1024) | 100M | 0.85 | 75 | 3 | 2.92 | 2.32 | 0.91 | 4.08 | 86.2 |
| DualCodec 12.5 Hz (16384+1024) | 95M | 0.80 | 75 | 6 | 2.94 | 2.33 | 0.91 | 4.04 | 85.2 |
| DualCodec 12.5 Hz (16384+4096) | 95M | 0.93 | 75 | 6 | 3.11 | 2.54 | 0.92 | 4.11 | 88.2 |

- Internal inconsistency in DualCodec: the text says the best model reaches MUSHRA 94.8, but Table 3 lists 88.2 for that model. — [DualCodec](https://arxiv.org/html/2505.13000)
- A reproduced DAC at 12.5 Hz, 0.75 kbps (6 VQ) reaches UTMOS 3.87 and MUSHRA 81.8. Much of the gain in "new" low-rate codecs comes from a low frame rate plus a large first codebook, not from exotic architecture. — [DualCodec](https://arxiv.org/html/2505.13000)

**Stable Codec / TAAE (Stability AI; arXiv 2411.19842v1, 29 Nov 2024, older), LibriSpeech test-clean 16 kHz** — [TAAE](https://arxiv.org/html/2411.19842)
- TAAE at 400 bps (1 token/frame, 25 tok/s): PESQ 2.96, STOI 0.90, SI-SDR 3.18, MOSNet 3.36.
- TAAE at 700 bps (2 tokens/frame, 50 tok/s): PESQ 3.09, STOI 0.92, MOSNet 3.36.
- Mimi in the same table: 1100 bps (8 cb) scores PESQ 3.01 and STOI 0.90; 550 bps scores PESQ 2.48.
- SemantiCodec in the same table: 337.5 bps scores PESQ 2.21 and STOI 0.79.
- TAAE has 950M parameters, against DAC 76M, Mimi 80M, and SemantiCodec 507M.
- The MUSHRA test (24 participants, 25 clips, no hidden anchor) appears only as a figure. The authors' claim is that TAAE "clearly outperforms" the other codecs and comes close to ground truth.

**TS3-Codec (arXiv 2411.18803, Nov 2024, older)** — a transformer-only streaming codec with a single codebook.
- Independent harness (VERSA, LibriSpeech test-clean): TS3-Codec at 0.85 kbps scores PESQ 2.23, UTMOS 3.84, and SPK-SIM 0.70. BigCodec at 1.04 kbps scores PESQ 2.68, UTMOS 4.11, and SPK-SIM 0.62 (snippet). — [VERSA](https://arxiv.org/pdf/2412.17667)
- TS3-Codec is not publicly released. FocalCodec's authors had to get reconstructions from the TS3 authors (snippet). — [search summary of FocalCodec / TS3 papers](https://arxiv.org/pdf/2411.18803)

**Other ~0.6–1.1 kbps points (2025–2026)**
- **NanoCodec** (NVIDIA, arXiv 2508.05835, Aug 2025, 22.05 kHz): its authors report that at 0.6 kbps it "significantly outperforms" TAAE on most metrics, though TAAE is slightly better on PESQ. They also report that their 0.8 kbps model is more intelligible than TS3-Codec at 0.85 kbps, with the caveat that TS3 runs at 16 kHz (snippet; the PDF tables could not be extracted). — [NanoCodec](https://arxiv.org/pdf/2508.05835)
- **CodecBench** (arXiv 2508.20660, Aug 2025): BigCodec performs best among single-codebook low-bitrate settings on speech, but underperforms Mimi-8 on music and sound (snippet). — [CodecBench](https://arxiv.org/pdf/2508.20660)
- **Multilingual X-Codec2 follow-up** (arXiv 2601.20185, v2 8 Mar 2026), with UTMOSv2 on Common Voice 17 (116 languages; English column shown):
  - NeuCodec (65536x1 @ 50 Hz): 2.361
  - LongCat (8192x4 @ 16.6 Hz): 2.341
  - The proposed 25 Hz / 24 kHz X-Codec2 variant: 2.245
  - Qwen3-TTS-Tokenizer (4096x16 @ 12 Hz): 2.236
  - WavTokenizer-40: 2.150
  - DistilCodec: 2.130
  - Original X-Codec-2.0: 2.086
  - SNAC: 2.048
  - Mimi-32: 2.032
  - EnCodec-2: 1.464

  The paper reports UTMOS only, so this table cannot be used to rank intelligibility. — [X-Codec2 multilingual](https://arxiv.org/html/2601.20185)
- **2026 newcomers at about 0.5–1.1 kbps (all self-reported):**
  - **VoCodec** (arXiv 2606.05892, Interspeech 2026): spends more bits on voiced frames than unvoiced ones, and reports beating baselines at 1.1 kbps on LibriTTS 16 kHz. — [VoCodec](https://arxiv.org/abs/2606.05892)
  - **PACodec** (arXiv 2609.03363, APSIPA 2026): replaces residual VQ with parallel additive VQ, and claims about 30% lower bitrate at the same quality. — [PACodec](https://arxiv.org/abs/2609.03363)
  - **CFMDCTCodec** (arXiv 2605.26812, TASLP): an MDCT-domain codec with a flow-matching enhancer, tested at 0.65 kbps on LibriTTS, where WavTokenizer scored about 73.8 MUSHRA (snippet). — [CFMDCTCodec](https://arxiv.org/pdf/2605.26812)
  - **ECC** (arXiv 2606.11631, 10 Jun 2026): adds explicit entropy models (scalar quantization plus a hyperprior and channel-wise context, with "entropy skip"). It claims average BD-rate savings of 39.9% on ViSQOL and 76.3% on PESQ over neural and conventional baselines. It is one of the few speech codecs to report entropy-coded rates. — [ECC RD benchmark](https://arxiv.org/abs/2606.11631)

### Inferences
- **Arithmetic checks on raw bitrates** (frame rate x bits per token):

  | Codec | Calculation | Raw rate |
  |---|---|---|
  | BigCodec | 80 Hz x 13 b | 1040 bps |
  | X-Codec2 | 50 Hz x 16 b | 800 bps |
  | BiCodec | 50 Hz x 13 b | 650 bps |
  | FocalCodec@50 / @25 / @12.5 | 50 / 25 / 12.5 Hz x 13 b | 650 / 325 / 162.5 bps |
  | Mimi | 12.5 Hz x 11 b x n codebooks | 687.5 bps (n=5), 1100 bps (n=8) |
  | Stable Codec | 25 Hz x 2 x log2(15625) ≈ 13.93 b | ≈ 697 bps |
  | TAAE-400 | 25 tok/s x 16 b | 400 bps |

- Mimi is listed at 0.55, 0.69, 0.83 or 1.1 kbps depending on how many RVQ levels a paper keeps, so always check n_q.
- BiCodec's 650 bps covers only its 50 Hz semantic stream. I recall from memory, unverified in this session, that it also sends a fixed-length set of global speaker tokens per utterance that the quoted rate leaves out.
- **Pareto set at about 0.6–1.1 kbps on LibriSpeech clean**, taking intelligibility and naturalness together:
  - FocalCodec@50 (0.65 kbps): best dWER of all, 2.18.
  - BiCodec (0.65 kbps): UTMOS 4.18, PESQ-WB 2.51.
  - X-Codec2 (0.8 kbps).
  - DualCodec 12.5 Hz (0.8–0.93 kbps): MUSHRA 85–88 in its own test.
  - BigCodec (1.04 kbps): highest PESQ-WB (2.68) and STOI (0.94) among single-codebook codecs, plus top Sim in the FocalCodec harness.
  - Stable Codec / TAAE has the highest UTMOS but noticeably worse dWER and Sim, so it is Pareto-optimal only on naturalness.
- **Above about 1.5 kbps** there is little new speech-specific research. DAC and EnCodec at 6 kbps with 8–12 codebooks are the reference points, and DAC-6k is near-transparent (PESQ-WB 4.01).

### Gaps
- I could not extract numeric MUSHRA values for TAAE, FocalCodec, or P2PSynCodec, because they appear only as figures.
- X-Codec2's own paper (arXiv 2502.04128) and its WER/SIM table were not fetched. Its numbers here come from the BiCodec paper's harness.
- SNAC, LSCodec, and Mimi's own MUSHRA results were not fetched as tables.
- The NanoCodec tables were not extractable because the PDF is compressed.
- No numbers were found for codecs in the 1.5–6 kbps band published in 2025–2026. The band seems to be neglected in research because quality there is already saturated.

---

## Q2. What is the frontier below 1 kbps and below 500 bps? How intelligible and speaker-faithful are these codecs really?

### Takeaway
Down to about 300–500 bps, single-codebook acoustic codecs still give good intelligibility on clean English: FocalCodec@25 reaches dWER 3.3 at 0.33 kbps. Around 150–250 bps, naturalness metrics stay high while intelligibility and speaker similarity begin to degrade: FocalCodec@12.5 keeps UTMOS 4.22 but has dWER 7.9 on English and 54% on multilingual. The sub-100 bps "tokenizers" (TaDiCodec at 87.5 bps) depend on a text transcript given to the decoder, so they are not stand-alone speech codecs. The usable frontier for a real, any-language, any-speaker codec currently bottoms out around 300–650 bps.

### Cited Findings
- **FocalCodec scaling (single 13-bit binary codebook, 16 kHz):**

  | Variant | Bitrate | UTMOS | dWER (EN) | Sim | dWER (MLS) |
  |---|---|---|---|---|---|
  | @50 | 650 bps | 4.05 | 2.18 | 97.4 | 12.57 |
  | @25 | 330 bps | 4.14 | 3.30 | 96.3 | 19.78 |
  | @12.5 | 160 bps | 4.22 | 7.94 | 93.9 | 54.15 |

  — [FocalCodec](https://arxiv.org/html/2502.04465)
- **Semantic k-means baseline WavLM6-KM (450 bps):** UTMOS 3.75 and dWER 6.20 on English, but dWER 44.54 on MLS. — [FocalCodec](https://arxiv.org/html/2502.04465)
- **WavTokenizer (480 bps, 40 Hz):** dWER 11.55 on English, 49.73 on MLS. — [FocalCodec](https://arxiv.org/html/2502.04465)
- **Streaming sub-kbps codecs (all 16 kHz):**
  - FocalCodec-Stream (arXiv 2509.16195, 19 Sep 2025): 0.55–0.80 kbps, single codebook, 80 ms theoretical latency. Its S@50-2k variant (0.55 kbps) scores UTMOS 3.88, dWER 4.63, Sim 99.4; the S@50-65k variant (0.80 kbps) scores UTMOS 3.85 and dWER 3.68. Mimi at 0.83 kbps (6 cb) scores UTMOS 3.44 and dWER 4.77. — [FocalCodec-Stream](https://arxiv.org/html/2509.16195)
  - ZipCodec (Della Libera, Subakan, Ravanelli; arXiv 2609.11642v1, 10 Sep 2026): 6.25 Hz frame rate, scalar spherical quantization of 64 dims x 4 levels (128 bits per frame, 0.80 kbps), 160 ms theoretical latency, and a streaming Vocos/iSTFT decoder. On LibriSpeech it scores UTMOS 3.89, dWER 2.83, Sim 98.0. On MLS it scores dWER 15.52, against 19.88 for FocalCodec-S and 26.12 for Mimi-0.83. This makes it the current best published streaming point near 0.8 kbps, with the caveat that the authors' own group made the baselines. — [ZipCodec](https://arxiv.org/html/2609.11642v1)
- **P2PSynCodec** (USTC/iFLYTEK; arXiv 2606.05876, 4 Jun 2026, Interspeech 2026):
  - Transmits one 1024-entry VQ at 50 Hz (0.5 kbps at 16 kHz). Three "pseudo-VQ" stages are predicted at the decoder and cost zero bits.
  - On LibriTTS it scores UTMOS 3.947, STOI 0.823, ViSQOL 3.476. BigCodec retrained to 0.5 kbps scores 3.939 / 0.872 / 3.682, and WavTokenizer at 0.5 kbps scores 3.269.
  - So P2PSynCodec matches BigCodec on UTMOS with 23M parameters against 159M (3.3 against 61 GFLOPs), but is lower on STOI and ViSQOL.
  - In the authors' ABX test (25 or more MTurk listeners), it was "comparable" to 1.5–2 kbps baselines. The paper reports no WER or SIM.

  — [P2PSynCodec](https://arxiv.org/html/2606.05876)
- **LSCodec** (Interspeech 2025): a speaker-decoupled codec at 0.45 and 0.25 kbps, evaluated with WER and SECS (snippet; table not fetched). — [LSCodec](https://www.isca-archive.org/interspeech_2025/guo25_interspeech.pdf)
- **FMelCodec** (Du, Ai, Jiang, Tian, Ling; arXiv 2605.25669, 25 May 2026, IEEE/ACM TASLP):
  - Mel-spectrogram codec with 640x compression and a single 1024-entry VQ.
  - A conditional flow-matching refiner sits in front of a HiFi-GAN vocoder.
  - Operating points: 250 bps at 16 kHz and 750 bps at 48 kHz.
  - Reported MOS for naturalness and speaker similarity is above 3.5 at 250 bps.
  - BigCodec had "markedly lower dWER at 250 bps" but was worse on the other quality metrics (snippet).

  — [FMelCodec](https://arxiv.org/abs/2605.25669)
- **AugCodec** (arXiv 2606.21893, Interspeech 2026): separate semantic, speaker, and prosody tokens with augmentation-driven disentanglement, at up to 12.5 Hz. Its 50 Hz variant matches FACodec-50Hz WER at about 60% of the bitrate. — [AugCodec](https://arxiv.org/abs/2606.21893)
- **Entropy-guided GRVQ** (arXiv 2603.01476, 2026): claims +21 MUSHRA over official Mimi in an ultra-low-bitrate setting, but the test had only 8 listeners and was run by the authors (snippet). — [GRVQ](https://arxiv.org/html/2603.01476v1)
- **SPG-Codec** (arXiv 2604.26296, 2026) studies the "role and boundaries of semantic priors" in ultra-low-bitrate coding. Its content was not retrieved. — [SPG-Codec](https://arxiv.org/pdf/2604.26296)
- **TaDiCodec** (CUHK-Shenzhen/Amphion; arXiv 2508.16790, NeurIPS 2025):
  - Diffusion autoencoder with a single-layer codebook at 6.25 Hz on 24 kHz speech, giving 0.0875 kbps (87.5 bps).
  - The diffusion decoder uses text guidance.
  - Trained end-to-end in a single stage.
  - Zero-shot TTS WER is 2.28 on SeedTTS test-en and 1.19 on test-zh, with RTF 0.12–0.29.

  — [TaDiCodec](https://arxiv.org/pdf/2508.16790)
- **Single-Codec** (304 bps, 23.4 tok/s): STOI 0.86, PESQ-WB 1.88, UTMOS 3.72, SIM 0.60. This is an older (2024) single-codebook point. — [Spark-TTS table](https://arxiv.org/html/2503.01710)
- **TAAE-400 bps:** PESQ 2.96 and STOI 0.90 on LibriSpeech. Of all the sub-500 bps points found, this is the best on signal fidelity, but it needs 950M parameters. — [TAAE](https://arxiv.org/html/2411.19842)

### Inferences
- **Approximate frontier as of Oct 2026, clean English, self-reported:**

  | Rate | Leading codecs | Typical quality |
  |---|---|---|
  | ~650–800 bps | FocalCodec@50, BiCodec, X-Codec2, ZipCodec (streaming) | dWER 2–3%, UTMOS ≥ 4 |
  | ~400–500 bps | TAAE-400, P2PSynCodec-0.5k | |
  | ~330 bps | FocalCodec@25 | dWER 3.3% |
  | ~160–250 bps | FocalCodec@12.5, FMelCodec-250 | dWER about 8% on English; poor on other languages |
  | < 100 bps | TaDiCodec | only with text side information |

- **Speaker fidelity:** the Sim reported by FocalCodec-family papers stays above 93 even at 160 bps. Sim on that scale is compressed near the top, though. In the BiCodec harness, the same class of codecs ranges 0.60–0.84, which shows larger gaps (see Q5).
- The steep jump in multilingual dWER at low rates (12.5 → 54%) suggests the ultra-low codecs work partly as English-phonetic recognizers plus a vocoder. On out-of-distribution languages they "hallucinate" plausible English-like phones.
- **TaDiCodec's 87.5 bps is not comparable to stand-alone codecs.** The decoder receives the transcript, so the real channel cost has to include the text, roughly 40–60 bps extra if entropy-coded (my estimate). It is a tokenizer for TTS/LM, not a communications codec.

### Gaps
- No entropy-coded bitrates were found for single-codebook sub-500 bps codecs. Token entropy is usually below log2(K), so real rates could be meaningfully lower.
- No independent sub-300 bps listening test was found.
- No numbers were found for "text/phoneme + prosody + speaker" semantic-communication schemes at about 50–150 bps, published 2025–2026 and evaluated as codecs with WER and SIM. Searches did not surface any.
- LSCodec numeric results were not fetched.

---

## Q3. How do generative decoders (diffusion, flow matching, LM-based) change the frontier compared with GAN decoders, and at what cost?

### Takeaway
Generative decoders mostly raise naturalness metrics at a fixed bitrate, and they keep speech clean down to about 250 bps. They do not improve intelligibility or speaker fidelity, and often make them worse. They also cost a lot of parameters, latency, and waveform alignment. Meanwhile, GAN-decoded single-codebook codecs (BigCodec, FocalCodec, X-Codec2, BiCodec, ZipCodec) have caught up to diffusion-decoded ones above about 300 bps.

### Cited Findings
- **SemantiCodec** (diffusion decoder, 0.31–1.40 kbps):
  - It is generative and "lacks precise temporal alignment," so SI-SDR cannot be computed for it. — [TAAE](https://arxiv.org/html/2411.19842)
  - At 0.65 kbps it scores only UTMOS 2.91 and dWER 8.97 against FocalCodec@50's 4.05 / 2.18 in the same harness. — [FocalCodec](https://arxiv.org/html/2502.04465)
  - It has 507M parameters. — [TAAE](https://arxiv.org/html/2411.19842)
  - Published range 0.31–1.40 kbps. — [SemantiCodec](https://arxiv.org/abs/2405.00233)
- **TAAE / Stable Codec** is a large transformer autoencoder (950M parameters), not diffusion. It has the highest UTMOS (4.32), above the clean reference of 4.09, yet higher dWER (4.97) and lower Sim (94.7) than BigCodec or FocalCodec, and 56.99 dWER multilingual. A UTMOS above the reference means the decoder produces "cleaner than real" speech, not faithful speech. — [FocalCodec](https://arxiv.org/html/2502.04465)
- **FocalCodec@12.5 UTMOS paradox:** UTMOS keeps rising as bitrate falls (4.05 → 4.14 → 4.22) while dWER rises (2.18 → 3.30 → 7.94). — [FocalCodec](https://arxiv.org/html/2502.04465)
- **Flow-matching 2026 codecs:**
  - FMelCodec: a conditional-flow-matching mel refiner plus HiFi-GAN. It claims better quality and speaker similarity with lower complexity than baselines at 250 bps, but BigCodec kept a markedly lower dWER at 250 bps (snippet). — [FMelCodec](https://arxiv.org/abs/2605.25669)
  - CFMDCTCodec: a flow-matching enhancer refines a coarse MDCT reconstruction. — [CFMDCTCodec](https://arxiv.org/abs/2605.26812)
- **TaDiCodec** (diffusion autoencoder, text-conditioned) reaches 87.5 bps. Its RTF is 0.12–0.29 in the TTS pipeline. — [TaDiCodec](https://arxiv.org/pdf/2508.16790)
- **Latency of non-generative streaming codecs:** FocalCodec-Stream 80 ms and ZipCodec 160 ms theoretical, with no diffusion steps. — [FocalCodec-Stream](https://arxiv.org/html/2509.16195), [ZipCodec](https://arxiv.org/html/2609.11642v1)
- **Decoder-side prediction as an alternative to generative decoding:** P2PSynCodec predicts 3 zero-bit "pseudo" VQ stages and reaches BigCodec-level UTMOS at 0.5 kbps with 23M parameters. — [P2PSynCodec](https://arxiv.org/html/2606.05876)

### Inferences
- **How generative decoding pays off:** it shifts the naturalness frontier down by roughly a factor of 2–4 in bitrate, so it sounds clean at 160–400 bps. The intelligibility and speaker frontier shifts much less.
- **Main costs:**
  - Content hallucination, visible as dWER growth and multilingual collapse.
  - Speaker drift, visible as Sim dropping toward 0.6 on stricter speaker models.
  - Loss of waveform alignment.
  - 5–10x more parameters.
  - Iterative sampling latency, which makes real-time streaming hard.
- **Naturalness-only metrics (UTMOS, MOSNet) systematically favor generative decoders.** Any frontier claim below about 400 bps should be checked against dWER (preferably multilingual) and an independent speaker model.

### Gaps
- No paper found directly quantifies "hallucination rate," such as substituted words or phones per minute, for generative codec decoders.
- No measured wall-clock latency or RTF comparisons were found for diffusion-decoded codecs against GAN-decoded codecs on the same hardware.

---

## Q4. What is known about fundamental limits, and how close are current systems?

### Takeaway
Linguistic content runs at about 39 bits/s across languages. Acoustic, information-theoretic estimates of the speech information that actually reaches listeners give an upper bound near 100 b/s. Current stand-alone codecs with good intelligibility run at about 300–650 bps raw (not entropy-coded), so they are roughly 3–15x above the content rate. They also still have to carry speaker identity and prosody, and no agreed bound exists for those.

### Cited Findings
- Across 17 languages, information rates converge to about 39.15 bits/s even though syllable rates differ (Coupé et al., Science Advances 2019). — [CNRS press release](https://www.cnrs.fr/en/press/similar-information-rates-across-languages-despite-divergent-speech-rates)
- **Van Kuyk, Kleijn & Hendriks (ICASSP 2017):**
  - Their method estimates the information rate of speech without relying on production or language models, and gives an upper bound of about 100 b/s.
  - Acoustic estimates are typically orders of magnitude larger than linguistic ones.
  - An earlier production-model estimate is about 1500 b/s.
  - Related: the SIIB intelligibility measure (bits/s shared between talker and listener).

  — [TU Delft repository](https://repository.tudelft.nl/file/File_94bfb808-583c-4ef2-84b9-91acc8394182), [Van Kuyk thesis](https://openaccess.wgtn.ac.nz/articles/thesis/Speech_Communication_from_an_Information_Theoretical_Perspective/17136377)
- Picone & Doddington (ICASSP 1989), on phonetic vocoding at about 100 bps: approaching 100 b/s implies approaching the true information rate of the communication process. — [Picone & Doddington 1989](https://isip.piconepress.com/publications/conference_proceedings/1989/ieee_icassp/vocoder/paper_v00.pdf)
- The best reported stand-alone points are FocalCodec@25 at 330 bps (dWER 3.3%) and FocalCodec@12.5 at 160 bps (dWER 7.9%). — [FocalCodec](https://arxiv.org/html/2502.04465)
- ECC (2026) shows that explicit entropy modeling can cut BD-rate by about 40% (ViSQOL) and about 76% (PESQ). Learned entropy coding has been "underexplored" in speech codecs. — [ECC](https://arxiv.org/abs/2606.11631)

### Inferences
- FocalCodec@12.5 at 160 bps sits only about 1.6x above the ~100 b/s acoustic bound and about 4x above the ~39 b/s linguistic rate. Its intelligibility loss (8% dWER English, 54% multilingual) suggests it is already pressing on the content limit when it lacks strong language priors.
- About 300–650 bps raw codecs are roughly 3–6x above the 100 b/s bound before any entropy coding.
- Applying entropy coding (likely 10–40% savings, extrapolating from ECC; unverified for token codecs) plus speaker and prosody factorization could plausibly reach about 150–250 bps with today's quality. That is an inference, not a reported result.

### Gaps
- No 2025–2026 paper was found that measures codec token entropy against these bounds.
- No accepted information-theoretic bound was found for the speaker-identity and prosody components.

---

## Q5. Which results hold up in independent comparisons, and which are only self-reported?

### Takeaway
A handful of rankings replicate across harnesses built by different groups: BigCodec, X-Codec2, BiCodec and FocalCodec beat EnCodec, DAC, SpeechTokenizer and WavTokenizer at about 1 kbps, and Mimi trails at matched sub-kbps rates. Absolute Sim and MUSHRA values do not transfer between papers. No independent multi-codec MUSHRA test was found, and almost every 2026 claim is self-reported.

### Cited Findings
- **BigCodec at about 1.04 kbps across harnesses:**

  | Harness | UTMOS | Speaker similarity |
  |---|---|---|
  | FocalCodec | 4.11 | 98.5 |
  | Spark-TTS | 4.11 | 0.84 |
  | VERSA | 4.11 | 0.62 (snippet) |

  UTMOS is stable, but speaker similarity varies widely with the embedding model. — [FocalCodec](https://arxiv.org/html/2502.04465), [Spark-TTS](https://arxiv.org/html/2503.01710), [VERSA](https://arxiv.org/pdf/2412.17667)
- **Stable Codec's UTMOS** is 4.32 in the FocalCodec harness and 4.23 in the Spark-TTS harness. Its SIM is 94.7 and 0.62 respectively. — [FocalCodec](https://arxiv.org/html/2502.04465), [Spark-TTS](https://arxiv.org/html/2503.01710)
- **Mimi** consistently underperforms at matched rates:
  - UTMOS 3.29 at 0.69 kbps (FocalCodec).
  - UTMOS 3.56 at 1.1 kbps (Spark-TTS).
  - MUSHRA 72.8 at 0.83 kbps against WavTokenizer's 81.0 (DualCodec).

  — [FocalCodec](https://arxiv.org/html/2502.04465), [Spark-TTS](https://arxiv.org/html/2503.01710), [DualCodec](https://arxiv.org/html/2505.13000)
- **TS3-Codec vs. BigCodec in VERSA:** TS3 wins on SPK-SIM (0.70 vs. 0.62), BigCodec wins on PESQ and UTMOS (snippet). TS3 is closed. — [VERSA](https://arxiv.org/pdf/2412.17667)
- **CodecBench** (independent): BigCodec is best among single-codebook codecs on speech (snippet). — [CodecBench](https://arxiv.org/pdf/2508.20660)
- **Listening tests are self-run, small, and differently designed:**
  - TAAE: 24 listeners, no anchor. — [TAAE](https://arxiv.org/html/2411.19842)
  - FocalCodec: 33 valid raters. — [FocalCodec](https://arxiv.org/html/2502.04465)
  - GRVQ: 8 listeners. — [GRVQ](https://arxiv.org/html/2603.01476v1)
  - P2PSynCodec: 25 or more MTurk listeners with a 3.5 kHz anchor. — [P2PSynCodec](https://arxiv.org/html/2606.05876)
  - DualCodec: internal MUSHRA inconsistency (94.8 in the text vs. 88.2 in the table). — [DualCodec](https://arxiv.org/html/2505.13000)
- **MUSHRA-1S** (arXiv 2509.19219) proposes a more sensitive MUSHRA variant, because standard MUSHRA struggles to separate top-tier systems. — [MUSHRA-1S](https://arxiv.org/pdf/2509.19219)
- **ZipCodec's and FocalCodec-Stream's baselines** all come from the same group (Della Libera, Subakan, Ravanelli), which re-uses one harness. That is internally consistent, but not independent. — [ZipCodec](https://arxiv.org/html/2609.11642v1)
- **Baseline retraining confound:** P2PSynCodec retrained BigCodec, DAC, and WavTokenizer to its own 0.5 kbps configuration. Baselines in that paper are therefore not the official checkpoints. — [P2PSynCodec](https://arxiv.org/html/2606.05876)

### Inferences
- **Robust (replicated in two or more independent harnesses):**
  - The single-codebook revolution: BigCodec, X-Codec2, BiCodec, FocalCodec and Stable Codec all score UTMOS ≥ 4.0 at 0.65–1.04 kbps, while EnCodec and DAC sit around 1.3–1.6 at about 1–1.5 kbps.
  - Mimi underperforms on reconstruction quality per bit.
  - TAAE / Stable Codec trades fidelity and speaker similarity for naturalness.
- **Self-reported only:** every 2026 claim, including ZipCodec, P2PSynCodec, VoCodec, PACodec, FMelCodec, CFMDCTCodec, AugCodec, GRVQ and ECC, plus NanoCodec's claim to beat TAAE and TS3.
- Cross-paper Sim must never be compared directly. UTMOS is the most portable metric, but it is biased toward generative output.

### Gaps
- No independent, multi-codec listening test covering X-Codec2, BiCodec, BigCodec, FocalCodec, TAAE and Mimi was found.
- Codec-SUPERB, DASB and VERSA leaderboards were not fetched as live tables, so it is unknown whether any 2026 leaderboard includes ZipCodec or P2PSynCodec.
