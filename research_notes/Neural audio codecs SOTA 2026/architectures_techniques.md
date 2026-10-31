# Neural Audio Codecs: Architectural and Algorithmic Techniques (state as of 2026-10-30)

Scope: design choices (quantization, frame rate, backbones, decoders, semantic distillation, entropy coding, objectives) and which have won, with ablation evidence. Leaderboards, the speech bitrate frontier, music/general audio and production standards are covered by other researchers.

Sourcing note: items marked **[fetched]** were read during this session (abstract, HTML or tables). Items marked **[prior knowledge, not re-fetched]** come from well-known papers I did not re-open in this session. The URL is the primary arXiv page, but the report writer should treat exact numbers in those items as needing a check. Dates are arXiv first-submission dates where known. Results from 2023–2024 are flagged as older.

---

## 1. Quantization: RVQ vs FSQ vs BSQ vs single large codebook vs continuous latents; codebook-collapse fixes

### Takeaway
Implicit-codebook scalar quantizers have largely won for new codecs, especially speech tokenizers built for LMs. These are FSQ, residual/grouped FSQ, and the binary/spherical variants (BSQ). They give about 95–100% utilization with no auxiliary losses, and they are more robust to bit errors. Classic RVQ survives where bitrate scalability or legacy compatibility matters (DAC/EnCodec/Mimi lineage, DualCodec). Practitioners rely on the "factorized low-dim + L2-normalized code lookup" fix (DAC) or SimVQ-style reparameterization to keep it healthy. The other major 2026 trend is to skip quantization entirely: low-frame-rate continuous VAE latents with continuous AR or flow-matching heads.

### Cited Findings
**Baselines and collapse fixes (older, 2021–2023)**
- SoundStream introduced RVQ for neural codecs, plus "quantizer dropout" (random truncation of the number of RVQ stages during training) so that one model serves several bitrates — [SoundStream, 2107.03312](https://arxiv.org/abs/2107.03312) [prior knowledge, not re-fetched; 2021].
- EnCodec uses RVQ with EMA codebook updates, k-means initialization and dead-code replacement, plus a loss balancer — [EnCodec, 2210.13438](https://arxiv.org/abs/2210.13438) [prior knowledge for the RVQ details; 2022].
- DAC's fix for collapse: factorized codes (look up in a low-dimensional, ~8-d, projected space) and L2-normalized codes. Together with quantizer dropout applied with probability 0.5 (not always), this gave near-full codebook use at 44.1 kHz / 8 kbps — [DAC, 2306.06546](https://arxiv.org/abs/2306.06546) [prior knowledge, not re-fetched; 2023]. DualCodec (2025) still uses exactly this recipe (D=8 projection + L2 norm, codebook and commitment losses). It names no other collapse fix, and it reports that larger codebooks help — [DualCodec, 2505.13000](https://arxiv.org/html/2505.13000v1) [fetched].
- SimVQ (Nov 2024; ICCV 2025) traces collapse to "disjoint optimization" of codebook entries: only the nearest codes get updated. It reparameterizes codes as a fixed coefficient matrix times one learnable linear basis, so the whole codebook moves together. On audio at about 0.9–1.35 kbps it reports 94.7–100% utilization regardless of codebook size — [SimVQ, 2411.02038](https://arxiv.org/html/2411.02038v1) [fetched via search summary]. SecoustiCodec (2025) ablates VQ vs SimVQ vs VAE-FSQ for a single-codebook streaming speech codec, and VAE-FSQ has the highest utilization (98.06%) — [SecoustiCodec, 2508.02849](https://arxiv.org/pdf/2508.02849) [search snippet].

**FSQ and residual FSQ**
- FSQ (Mentzer et al., 2023) bounds each latent dimension and rounds it to L levels, giving an implicit product codebook with no codebook loss or commitment loss — [FSQ, 2309.15505](https://arxiv.org/abs/2309.15505) [prior knowledge].
- TAAE / Stable Codec (Stability AI, Nov 2024) uses a 6-dim FSQ bottleneck. During training the levels per dimension are drawn at random from {17, 9, 5}, and training mixes uniform-noise emulation with straight-through gradients 50/50. After training, the levels can be changed, or residual FSQ used (for levels of the form 2^n+1), without retraining. Inference variants: one token per frame with a 46,656 codebook (400 bps at 25 Hz), or two residual tokens per frame with 15,625 per stage (700 bps) — [TAAE, 2411.19842](https://arxiv.org/html/2411.19842) [fetched].
- TAAE utilization (normalized entropy), from the same paper's table: FSQ 0.97 (400 bps) and RFSQ 0.97/0.95, vs DAC RVQ-1..4 at 0.87/0.97/0.97/0.99, EnCodec 0.78–0.86, Mimi 0.91–0.92, SpeechTokenizer 0.98/0.95/0.95 — [TAAE, 2411.19842](https://arxiv.org/html/2411.19842) [fetched].
- FSQ robustness (Sep 2025): FSQ codecs (NeuCodec, Stable Codec) were compared with RVQ codecs (DAC, EnCodec). FSQ reached almost complete codebook use with no auxiliary losses and had "vastly superior" bit-flip / noisy-channel robustness, because neighboring levels are perceptually close while RVQ indices are arbitrary — [FSQ for redundant, transmission-robust coding, 2509.09550](https://arxiv.org/pdf/2509.09550) [search summary]. The "redundant" framing means FSQ codes degrade gracefully.
- XCodec2 (Feb 2025, from the Llasa paper) uses a single FSQ codebook of 65,536 (4^8) at 50 Hz / 16 kHz, about 0.8 kbps, and fuses w2v-BERT-2.0 semantic features with an acoustic encoder — [Llasa/XCodec2, 2502.04128](https://arxiv.org/abs/2502.04128) [prior knowledge for config]. NVIDIA's Low Frame-rate Speech Codec uses FSQ at 21.5 Hz (about 1.9 kbps, 8 FSQ codebooks) — [LFSC, 2409.12117](https://arxiv.org/pdf/2409.12117) [prior knowledge for config; 2024].
- Factorized FSQ for LMs (Jun 2026): the d dimensions are split into n groups. Each group is predicted as a base-L index (per-group vocab 4096 = 2^12), and a 2-layer NAR "audio head" models dependencies between groups. This keeps the LM softmax tractable at about 288 bits/frame. At 4.17 Hz, TTS WER is 1.83/1.90 with the NAR head vs 10.17/12.73 with a linear head — [Speech-text alignment on frame rate, 2606.12199](https://arxiv.org/html/2606.12199) [fetched]. This is the main remaining cost of FSQ: a huge implicit vocabulary has to be factorized for the LM.

**Binary / spherical quantization (BSQ)**
- BSQ (Jun 2024, vision) projects to a low-dim unit hypersphere and binarizes each dimension. This gives an implicit 2^L codebook with bounded quantization error and entropy regularization — [BSQ, 2406.07548](https://arxiv.org/abs/2406.07548) [prior knowledge].
- FocalCodec (NeurIPS 2025) brings BSQ to speech. It uses WavLM features, focal-modulation compressor/decompressor blocks and a **single binary codebook** at 0.16–0.65 kbps. FocalCodec-Stream runs at 0.55–0.80 kbps with 80 ms theoretical latency — [FocalCodec, 2502.04465](https://arxiv.org/html/2502.04465v2); [NeurIPS 2025 abstract](https://proceedings.neurips.cc/paper_files/paper/2025/hash/220e6dbed88faa83bd9f9e8fa359e43d-Abstract-Conference.html) [search summary; BSQ detail from a secondary review plus the abstract's "binary codebook"].
- Q2D2 (Dec 2025) proposes geometry-aware 2-D quantization. It argues that RVQ/VQ "are challenging to optimize, and lead to the well-documented problem of underutilized codebooks", while FSQ "avoids collapse by design" — [Q2D2, 2512.01537](https://arxiv.org/pdf/2512.01537) [search snippet].

**Single large codebook vs multi-codebook**
- WavTokenizer (Aug 2024) uses one VQ codebook (4096) at 40 or 75 tokens/s at 24 kHz. Utilization comes from k-means init, random restart, a larger contextual window and an attention-augmented Vocos-style decoder — [WavTokenizer, 2408.16532](https://arxiv.org/abs/2408.16532) [prior knowledge for details]. BigCodec (Sep 2024) uses a single 8192 VQ at 80 Hz (1.04 kbps) with low-dim factorized, L2-normalized codes and LSTM layers in the encoder and decoder — [BigCodec, 2409.05377](https://arxiv.org/abs/2409.05377) [prior knowledge].
- A survey notes that single-layer codecs suit AR LM modeling, while multi-layer tokens can be generated in parallel by downstream LMs (delay/depth transformers) — [UniTok-Audio, 2510.26372](https://arxiv.org/pdf/2510.26372) [search snippet]. The TMLR 2025 survey/benchmark gives a taxonomy (quantizer, domain, sampling rate, single vs multi-domain training) and benchmarks speech, music and general audio — [Discrete Audio Tokens: More Than a Survey!, 2506.10274](https://arxiv.org/abs/2506.10274) [search summary].

**Continuous latents (no quantization)**
- Locodec / MP-ELD (Jul 31, 2026) uses 8 Hz, 768-dim continuous tokens with an autoregressive flow-matching generator. It reports good reconstruction and stable long-form synthesis without external SSL/ASR models — [2607.29363](https://arxiv.org/abs/2607.29363) [search summary].
- For transmission, continuous latents need a learned entropy model (hyperprior-style). I found no 2025–2026 audio-specific paper with a clean RVQ vs entropy-coded-continuous comparison at matched bitrate (see Gaps).

### Inferences
- The field has moved away from explicit codebooks plus auxiliary losses (EMA, commitment, dead-code restarts) toward implicit codebooks (FSQ/BSQ) for any new from-scratch codec. RVQ persists where bitrate scalability through quantizer dropout is wanted, or where models must stay compatible with DAC/EnCodec/Mimi ecosystems.
- Once collapse is fixed (by DAC's factorized + L2 trick, SimVQ or FSQ), utilization is about 0.95+ for all of them. Utilization alone no longer separates quantizers. The differentiators are LM-friendliness (vocab factorization), robustness to channel errors, and post-hoc bitrate flexibility.
- The 2026 split: compression-oriented codecs stay discrete (FSQ/RVQ). Generation-oriented tokenizers are increasingly continuous (VAE at 8–12.5 Hz plus a flow/diffusion head) or discrete with very large factorized FSQ vocabularies.

### Gaps
- No controlled ablation found that holds encoder, decoder and data fixed and swaps only RVQ / FSQ / BSQ / SimVQ, measuring both reconstruction and downstream LM quality. The comparisons available vary several factors at once (noted explicitly by the 2509.09550 summary).
- NDVQ (normal-distribution VQ) was not investigated in this session.
- The MOSS-Audio-Tokenizer quantizer type (RVQ vs FSQ) was not confirmed; the abstract does not state it.

---

## 2. Frame rate and token rate (12.5 / 25 Hz and below)

### Takeaway
The default has moved from 50–86 Hz (EnCodec/DAC) to 12.5–25 Hz (Mimi 12.5 Hz, TAAE 25 Hz, DualCodec 12.5/25 Hz), and research tokenizers in 2026 go to 2–8 Hz or variable rates. Reconstruction can survive very low frame rates if bits per frame rise. For LMs, there is a sweet spot near the text token rate (about 3–6 Hz for English). Below it TTS intelligibility collapses, and above it sequence length hurts reasoning. 25 Hz still beats 12.5 Hz on TTS WER in controlled codec ablations.

### Cited Findings
- Older reference points: EnCodec at 75 Hz (24 kHz), DAC at 86 Hz (44.1 kHz) — [EnCodec](https://arxiv.org/abs/2210.13438), [DAC](https://arxiv.org/abs/2306.06546) [prior knowledge]. Mimi at 12.5 Hz with 8 RVQ levels, about 1.1 kbps — [Moshi, 2410.00037](https://arxiv.org/abs/2410.00037) [prior knowledge for config; Sep 2024]. TAAE (Mimi row in its table: 1100 bps) is consistent.
- Matched frame-rate study (Jun 10, 2026). The setup used a fixed 600 bps information rate and a frozen Qwen3-4B, testing 50 down to 2.08 Hz. ASR WER was U-shaped, best at about 12.5–4.17 Hz. TTS WER worsened monotonically as frame rate fell and degraded sharply once speech became shorter than text (about 3.32 tokens/s on LibriSpeech). Speech-to-speech QA peaked at 4.17–6.25 Hz and was poor at 50 Hz. The LLM tolerated a speech-to-text length ratio of about 0.8×–2.2×. Reconstruction at 4.17 Hz (WER 3.37, UTMOS 3.79) was close to 25 Hz (3.04, 3.71) and beat CosyVoice2's 25 Hz tokens (4.10, 3.65) — [2606.12199](https://arxiv.org/html/2606.12199) [fetched].
- In DualCodec's controlled 25 vs 12.5 Hz comparison at about 75 tokens/s, TTS WER for VALL-E was 3.40 at 25 Hz vs 4.40 at 12.5 Hz, and for AR+SoundStorm 3.56 vs 4.93. RTF roughly halved at 12.5 Hz (0.30→0.16 and 0.66→0.34). Reconstruction UTMOS was slightly *higher* at 12.5 Hz (3.94 vs 3.86) because it had more bits per frame. The paper notes a remaining 12.5-vs-25 Hz TTS gap — [DualCodec, 2505.13000](https://arxiv.org/html/2505.13000v1) [fetched].
- When TAAE was pushed from 25 Hz to 12.5 Hz (d=8, 200 bps), quality dropped (PESQ 2.34, STOI 0.82, vs 2.96 / 0.90 at 400 bps, 25 Hz) and the model converged more slowly — [TAAE](https://arxiv.org/html/2411.19842) [fetched].
- SiTok (Feb 2026) is a 1.6B diffusion-autoencoder speech tokenizer at 12.5 Hz / 200 bps, trained on 2M hours. Its authors claim it beats baselines on understanding, reconstruction and generation — [SiTok, 2602.06602](https://arxiv.org/abs/2602.06602) [fetched abstract; no numbers].
- Variable or dynamic frame rate: FlexiCodec (ICLR 2026) averages below 10 Hz with a flow-matching decoder — [2510.00981](https://arxiv.org/pdf/2510.00981). DyCAST does character-aligned variable-rate tokenization — [2601.23174](https://arxiv.org/html/2601.23174v1). U-Codec is an ultra-low frame-rate codec — [2510.16718](https://arxiv.org/pdf/2510.16718). FlexiSLM uses controllable frame rates — [2606.31247](https://arxiv.org/html/2606.31247v3). Locodec runs at 8 Hz continuous — [2607.29363](https://arxiv.org/abs/2607.29363) [all search summaries].

### Inferences
- Frame rate and bits per frame trade off. For pure compression, lower frame rates with more bits per frame are roughly neutral or slightly better (more context per latent, as in DualCodec 12.5 Hz UTMOS). For LMs, lower rates cut sequence length and speed up generation (about 2× RTF from 25 to 12.5 Hz), but each step's prediction gets harder. That is why factorized-FSQ and NAR or flow heads appear together with sub-10 Hz rates.
- The 2026 direction is content-adaptive (variable) rates near the text rate for spoken-LM use, while fixed 12.5–25 Hz remains the practical standard for codecs.

### Gaps
- No apples-to-apples study was found of how low frame rate affects music or general audio (the matched study is English read speech only).

---

## 3. Encoder/decoder backbones

### Takeaway
Convolutional SEANet/DAC-style encoders and decoders (strided convs, Snake activations) remain the workhorse. The scaled frontier, however, is transformer-heavy or transformer-only. TAAE (950M) showed transformers with patching beat conv codecs at very low bitrates. In 2026, the 1.6B MOSS-Audio-Tokenizer / CAT and TAC (ICML 2026) report predictable scaling of end-to-end transformer tokenizers trained on about 3M hours. Vocos-style iSTFT heads are the standard for cheap, low-frame-rate decoders, at the cost of slight phasiness. I found no evidence that Mamba/SSM backbones have displaced either.

### Cited Findings
- TAAE architecture: 320-sample patching at 16 kHz, 8 transformer blocks, then 2× downsample, then 20 blocks. Dim 1024, sliding-window attention (128), RoPE, QK-norm, LayerScale, gated SiLU MLPs, LN ε=1e-2, mirrored decoder, about 950M params total. Trained on about 105k h on 16 H100s. Scale ablation at 700 bps: 240M PESQ 2.74, 540M 2.80, 950M 2.82, so scaling yields only small gains in that range. RTF on H100 is 0.0024 for 30 s clips despite the size — [TAAE, 2411.19842](https://arxiv.org/html/2411.19842) [fetched].
- TAAE at 400/700 bps vs conv/RVQ baselines (LibriSpeech test-clean): PESQ 2.96/3.09 vs Mimi 3.01 @1100 bps, DAC 2.29 @2000 bps, EnCodec 2.84 @3000 bps. It was significantly ahead in MUSHRA — [TAAE](https://arxiv.org/html/2411.19842) [fetched].
- TS3-Codec (Nov 2024) is a transformer-only streaming codec. It reports beating conv codecs such as BigCodec at lower compute — [TS3-Codec, 2411.18803](https://arxiv.org/abs/2411.18803) [prior knowledge, not re-fetched].
- MOSS-Audio-Tokenizer (Feb 2026) has a purely transformer CAT architecture: encoder, quantizer and decoder trained jointly from scratch with no pretrained encoders. It has 1.6B params, about 3M hours of general audio, and "predictable improvements with increased scale". The authors claim it beats prior codecs across bitrates on speech, sound and music, and enables a pure-AR TTS that surpasses NAR/cascaded systems — [MOSS-Audio-Tokenizer, 2602.10934](https://arxiv.org/abs/2602.10934) [fetched abstract]. A search summary says the paper ties reconstruction fidelity to training batch size (not verified).
- TAC, "Scaling Transformers for End-to-End Discrete Audio Tokenization" (ICML 2026, PMLR v306), reports beating EnCodec, DAC and Mimi on speaker similarity and subjective quality across bitrates, with predictable scaling — [PMLR](https://proceedings.mlr.press/v306/gong26c.html) [search summary]. It may be the same work as MOSS/CAT or a sibling; not verified.
- Mimi adds transformer layers to a SEANet conv encoder and decoder (hybrid) — [Moshi, 2410.00037](https://arxiv.org/abs/2410.00037) [prior knowledge]. BigCodec adds LSTMs to conv stacks — [2409.05377](https://arxiv.org/abs/2409.05377) [prior knowledge].
- Vocos predicts complex STFT coefficients and synthesizes with an iSTFT (ConvNeXt backbone, no transposed-conv upsampling). It matches SOTA quality with roughly an order-of-magnitude speedup over time-domain vocoders — [Vocos, 2306.00814](https://arxiv.org/pdf/2306.00814) [search summary; ICLR 2024]. A 2026 IWAENC paper finds Vocos quality "slightly below the state-of-the-art time-domain vocoders, with artifacts resembling phasiness" — [IWAENC 2026, AudioLabs Erlangen](https://audiolabs-erlangen.de/content/resources/NLUI/00_2026-IWAENC-vocoder/main.pdf) [search summary]. XCodec2 ships a Vocos-style decoder — [HKUSTAudio/xcodec2 codec_decoder_vocos.py](https://huggingface.co/HKUSTAudio/xcodec2/blob/11ff96fa3ac76e1eee69da30dcf51163fdb1c330/vq/codec_decoder_vocos.py). WavTokenizer uses an attention-augmented Vocos decoder — [2408.16532](https://arxiv.org/abs/2408.16532) [prior knowledge].
- BigVGAN-style decoders (Snake periodic activations, anti-aliased multi-periodicity) are the time-domain alternative. DAC adopted Snake activations — [BigVGAN, 2206.04658](https://arxiv.org/abs/2206.04658); [DAC](https://arxiv.org/abs/2306.06546) [prior knowledge].
- Band-split designs persist for universal audio (BSCodec) — [2511.06150](https://arxiv.org/pdf/2511.06150) [search snippet].
- Mamba/SSM: the search found no codec paper comparing an SSM backbone with conv/transformer/Vocos decoders. The only Mamba hit was a TTS system (MambaVoiceCloning, 2604.00292), not a codec — [2604.00292](https://arxiv.org/pdf/2604.00292).

### Inferences
- "Transformer-only plus scale plus FSQ" (TAAE, then MOSS/CAT and TAC) is the 2025–2026 frontier recipe for tokenizers. Conv-based DAC-style models remain dominant for lightweight, low-latency and on-device codecs because they are cheap per sample.
- iSTFT heads win when the decoder must upsample a lot from 12.5–25 Hz cheaply. Time-domain BigVGAN-style decoders still win on peak fidelity.

### Gaps
- Mamba/SSM codecs: no reliable evidence found on whether they help. Treat as unresolved or niche.
- The MOSS/CAT and TAC papers' internal ablations (batch-size scaling, quantizer) were not read; only abstracts.

---

## 4. Decoders: GAN vs diffusion/flow-matching vs LM-based token decoders; perception vs fidelity

### Takeaway
GAN decoders (multi-scale STFT and multi-period discriminators with feature matching) remain the default for real-time codecs. At very low bitrates (≤ about 1 kbps), and in 2026 scaled tokenizers, diffusion or flow-matching decoders are increasingly used because they can "sample what you can't compress". They trade off fidelity and need several steps, typically distilled. Mimi showed that dropping reconstruction losses and training only adversarially improves perceived quality while hurting objective metrics, which is a practical demonstration of the perception–distortion tradeoff.

### Cited Findings
- Discriminator standard. EnCodec uses a multi-scale STFT (MS-STFT) discriminator. DAC uses a multi-period discriminator plus a multi-band, multi-scale complex-STFT discriminator with feature matching. TAAE uses a multi-resolution complex-STFT discriminator ensemble with hinge loss, and its generator uses *only* normalized feature-matching L1 (no direct adversarial term) — [EnCodec](https://arxiv.org/abs/2210.13438), [DAC](https://arxiv.org/abs/2306.06546) [prior knowledge]; [TAAE](https://arxiv.org/html/2411.19842) [fetched].
- Mimi/Moshi trained with adversarial and feature-matching losses only (no reconstruction loss). Subjective quality improved even though objective metrics got worse — [Moshi, 2410.00037](https://arxiv.org/abs/2410.00037) [prior knowledge; exact MUSHRA deltas not re-verified].
- Multi-Band Diffusion (2023) decodes EnCodec tokens with a diffusion model, rated up to 30% better than EnCodec on speech and on par on music. It notes "blurred" high frequencies — [MBD, 2308.02560](https://arxiv.org/pdf/2308.02560) [search summary; older].
- FlowDec (Mar 2025) is a flow-based full-band general audio codec, on par with GAN-based DAC in listening tests at 4.5–7.5 kbps — [FlowDec, 2503.01485](https://arxiv.org/html/2503.01485) [search summary].
- Flow-matching decoders are standard in 2025–2026 low-frame-rate tokenizers: FlexiCodec (ICLR 2026), Locodec (Jul 2026), and SiTok's diffusion autoencoder at 12.5 Hz / 200 bps — [2510.00981](https://arxiv.org/pdf/2510.00981), [2607.29363](https://arxiv.org/abs/2607.29363), [2602.06602](https://arxiv.org/abs/2602.06602).
- Generic diffusion-decoder findings: inference cost is the main limitation, which fewer steps or distillation can mitigate. In image codecs, higher perception can harm fidelity in semantically important regions — [Sample what you can't compress, 2409.02529](https://arxiv.org/pdf/2409.02529); [CVPR 2025 region-adaptive diffusion](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.pdf) [search summaries].
- LM-based token decoders: LMCodec (2023) losslessly transmits coarse SoundStream tokens and lets a causal transformer generate the fine RVQ levels — [LMCodec, 2303.12984](https://arxiv.org/pdf/2303.12984) [search summary; older].
- Rate–distortion–perception theory: at a fixed rate, perfect perceptual quality (output distribution equal to data distribution) costs at most 2× MSE, and achieving it can require stochastic decoders — [Blau & Michaeli, 1901.07821](https://arxiv.org/abs/1901.07821) [prior knowledge]. This motivates generative decoders at very low rates.

### Inferences
- Rule of thumb as of 2026: at ≥ about 1.5 kbps, GAN or feature-matching decoders are fine and fast. Below about 1 kbps (and for semantic-only tokens), generative decoders (flow, diffusion, or LM) are needed to fill in the detail the rate cannot carry. Hybrid pipelines (semantic tokens → flow-matching → mel → GAN vocoder, as in CosyVoice-type stacks) are common.
- Diffusion and flow decoders let you trade perception for distortion at test time (steps or guidance). GAN decoders fix the tradeoff at training time.

### Gaps
- No controlled 2025–2026 head-to-head of GAN vs flow decoders on the same audio tokenizer was found. The available comparisons are cross-system.

---

## 5. Semantic distillation and disentanglement

### Takeaway
Some form of semantic supervision has clearly won for codecs meant for speech LMs. It cuts TTS WER by large factors. Variants include SSL distillation into the first codebook (SpeechTokenizer, Mimi), SSL feature fusion before quantization (X-Codec, XCodec2, DualCodec), and SSL encoders (FocalCodec on WavLM). The cost is reconstruction quality in the shared codebook, which is why designs moved from "distill into RVQ-1" to separate or parallel semantic branches. The leading 2026 counter-trend argues for end-to-end scaled training with no pretrained encoders or semantic distillation (MOSS-Audio-Tokenizer, TAC).

### Cited Findings
- Mimi distills WavLM into a separate semantic VQ in parallel with a 7-level acoustic RVQ ("split RVQ"). This is causal while matching non-causal WavLM targets. The split arose because distilling into the first level of a single RVQ trades phonetic discriminability against reconstruction — [Moshi, 2410.00037](https://arxiv.org/abs/2410.00037) [prior knowledge, corroborated by search summaries of the survey and WavChat]. Overweighting Mimi's semantic stream raises sBLIMP from 52.25% to 60.17% — [Discrete Audio Tokens survey, 2506.10274](https://arxiv.org/pdf/2506.10274) [search snippet].
- Distillation "significantly enhances the speech discrimination capability of the first quantizer; however, it can also negatively impact speech quality" — [search summary citing 2603.10371](https://arxiv.org/html/2603.10371v1) [snippet; context not verified].
- DualCodec ablation (Seed-TTS-Eval WER of RVQ-1 tokens, EN/ZH): DAC 25 Hz 55.4/46.4; plus distillation 28.4/26.4; plus dual encoding (w2v-BERT-2.0 layer-16 SSL branch → RVQ-1, waveform branch → residual) 5.59/6.52; plus a 16,384-entry RVQ-1 2.98/2.91 (ground truth 2.13/1.25). Downstream VALL-E TTS: DualCodec 25 Hz WER 3.40 / SIM 0.57 vs Mimi 8.16/0.48 and SpeechTokenizer 15.4/0.47. Reconstruction also improved: UTMOS 3.86 vs DAC-repro 3.29, MUSHRA 79.5 vs 68.8 at 25 Hz — [DualCodec, 2505.13000](https://arxiv.org/html/2505.13000v1) [fetched]. (The paper's text claims a MUSHRA of 94.8 for the best model, but its Table 3 gives 88.2, an internal inconsistency.)
- X-Codec (AAAI 2025) concatenates pre-trained semantic features before RVQ and adds a semantic reconstruction loss after RVQ. It improves zero-shot TTS WER over DAC/EnCodec and beats SpeechTokenizer on WER, Sim-O and UTMOS — [X-Codec, 2408.17175](https://arxiv.org/html/2408.17175v3) [search summary].
- FACodec (NaturalSpeech 3, 2024) factorizes content, prosody, timbre and acoustic detail with supervision and gradient reversal — [NaturalSpeech 3, 2403.03100](https://arxiv.org/abs/2403.03100) [prior knowledge; older].
- "Reconstruct! Don't Encode" (Mar 2026) reconstructs SSL representations from the codec as a loss rather than encoding them, aiming at high intelligibility with low latency for streaming — [2603.05887](https://arxiv.org/pdf/2603.05887) [title/snippet only].
- Counter-trend: MOSS-Audio-Tokenizer explicitly argues against semantic distillation and pretrained encoders in favor of fully end-to-end training at scale (1.6B, 3M h). Its authors still claim competitive ASR without auxiliary encoders — [2602.10934](https://arxiv.org/abs/2602.10934) [fetched abstract]. EntangleCodec (Jun 2026) argues for semantic–acoustic *entanglement* rather than disentanglement — [2606.02739](https://arxiv.org/html/2606.02739) [title/snippet].

### Inferences
- Semantic supervision mainly helps *generation* (LM-predictability, intelligibility). For pure compression it is neutral to negative unless placed in a parallel branch, as DualCodec shows both can improve when the branches are separated.
- The open 2026 question is whether scale substitutes for distillation (MOSS/TAC claim it does). This is not yet independently confirmed.

### Gaps
- No independent head-to-head of X-Codec vs XCodec2 vs DualCodec vs Mimi under a common TTS model was found.

---

## 6. Entropy coding of tokens (learned entropy models, LM-based arithmetic coding)

### Takeaway
For well-utilized modern codecs, static entropy coding of tokens saves only about 3–10%, because utilization is already about 0.9+. Context-adaptive transformer priors save more: EnCodec reported up to 40% (2022). LM-as-compressor results on raw waveforms beat FLAC but give diminishing gains at higher bit depth. The current bitrate for any codec should be stated as nominal (log2 of codebook) unless an entropy coder is specified.

### Cited Findings
- EnCodec: "lightweight Transformer models can be used to further compress the obtained representation by up to 40%", staying faster than real time — [EnCodec abstract, 2210.13438](https://arxiv.org/abs/2210.13438) [fetched; 2022].
- Huffman (unigram, static) coding of tokens, nominal → coded bps, from TAAE's table: EnCodec 3000→2510 (−16%), DAC 2000→1900 (−5%), SpeechTokenizer 1500→1450 (−3%), Mimi 1100→1005 (−9%), SemantiCodec 675→625 (−7%), TAAE 400→378 (−5.5%), 700→670 (−4%). The authors note that EnCodec gains most because its distribution is skewed — [TAAE, 2411.19842](https://arxiv.org/html/2411.19842) [fetched].
- LMCodec (2023) entropy-codes coarse tokens under a predicted distribution (Huffman in experiments, arithmetic coding noted as an option) — [2303.12984](https://arxiv.org/pdf/2303.12984) [search summary; older].
- LMs as lossless compressors of full-fidelity audio (Interspeech 2026; Long, Novack, Donahue). Autoregressive LMs on raw waveforms (music, speech, bioacoustics; 16–48 kHz; 8/16/24-bit) "consistently outperform FLAC" and reach SOTA at 8 and 16 bits, but gains shrink above 8-bit. Their "Trilobyte" byte-level tokenization makes 24-bit tractable (O(1) instead of O(2^b) vocab) — [2603.08683](https://arxiv.org/abs/2603.08683) [fetched abstract; ratios not on page]. The paper argues per-token log-likelihood predicts compression rate — [same, search summary].
- A secondary citation: a 70B transformer compressed LibriSpeech to about 16.4% of raw size vs about 30.3% for FLAC (likely Delétang et al., "Language Modeling Is Compression", 2023) — [cited in STCTS, 2512.00451](https://arxiv.org/pdf/2512.00451) [secondary; verify against the original].

### Inferences
- For a codec with a TAAE-style about 0.97 normalized entropy, the *unigram* headroom is only about 3–5%. Real gains require conditional (context) modeling of tokens across time and RVQ levels. That is exactly what a downstream speech LM already learns. So a codec plus an LM prior can approach the LM's cross-entropy (bits/token) as the real bitrate. This is plausibly the 25–40% range EnCodec reported for redundant tokens, and much less for compact 12.5–25 Hz tokenizers.

### Gaps
- No 2025–2026 paper found that measures arithmetic coding of modern low-rate tokens (Mimi, XCodec2, TAAE) under a strong transformer prior and reports bits saved. This is a clear gap. Continuous-latent audio codecs with hyperprior entropy models (image-codec style) were not found in this session.

---

## 7. Training objectives

### Takeaway
The stable recipe is multi-resolution mel/STFT L1, plus adversarial loss (hinge or LS) with multi-scale/multi-period STFT discriminators, plus discriminator feature matching. On top of this, quantizer losses (or none for FSQ), and increasingly an SSL-feature perceptual loss (WavLM) and/or semantic distillation. Perceptual SSL losses give large gains in perceptual metrics at small cost to SI-SDR. Aggressive perception-only training (Mimi, TAAE finetuning) is the practical application of rate–distortion–perception theory.

### Cited Findings
- DAC's multi-scale mel loss plus MPD/MRD-style discriminators plus feature matching (the standard recipe) — [DAC, 2306.06546](https://arxiv.org/abs/2306.06546) [prior knowledge]. EnCodec's gradient-normalizing loss balancer — [EnCodec](https://arxiv.org/abs/2210.13438) [prior knowledge].
- TAAE two-stage objective. Pretraining (500k steps) uses waveform L1 plus STFT-magnitude L1 that *decay* (γ=0.9999 per step) alongside feature matching. Finetuning (150k steps) replaces them with a WavLM-Large perceptual feature loss. Ablation at 700 bps, with vs without the WavLM loss: PESQ 3.09 vs 2.82, STOI 0.92 vs 0.88, Mel 0.86 vs 1.18, SI-SDR 4.73 vs 4.80 (nearly flat) — [TAAE, 2411.19842](https://arxiv.org/html/2411.19842) [fetched].
- NVIDIA LFSC uses a speech-LM (SSL-based) discriminator in addition to waveform/STFT discriminators — [LFSC, 2409.12117](https://arxiv.org/pdf/2409.12117) [prior knowledge; 2024].
- SSL-representation reconstruction as a loss for streaming intelligibility — [2603.05887](https://arxiv.org/pdf/2603.05887) [snippet].
- Rate–distortion–perception theory (Blau & Michaeli 2019) gives the formal justification for distribution-matching (adversarial or generative) losses at low rate — [1901.07821](https://arxiv.org/abs/1901.07821) [prior knowledge]. Codec practice: GAN losses are "typically added to the rate-distortion loss", while diffusion decoders can move along the perception–distortion curve at inference — [FlowDec, 2503.01485](https://arxiv.org/html/2503.01485); [Diffusion decoders overview](https://www.emergentmind.com/topics/diffusion-decoders) [search summaries; the overview is an aggregator].

### Inferences
- The 2026 recipe has shifted from fidelity (L1, mel) toward perceptual or semantic feature-space losses (WavLM perceptual loss, SSL reconstruction, SLM discriminators), especially below 1 kbps, where waveform-matching losses mainly add muffling.

### Gaps
- No systematic cross-paper ablation of loss weightings (mel vs feature matching vs perceptual SSL) under a fixed architecture was found. TAAE's single ablation is the cleanest evidence located.
