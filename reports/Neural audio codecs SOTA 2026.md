# Neural speech codecs break the kilobit barrier

As of late October 2026, the best neural speech codecs match clean reference speech on naturalness at about **0.65–1 kbps** and keep word error close to the original. The codecs that do this are single-codebook 16 kHz models: BigCodec at 1.04 kbps, X-Codec2 at 0.8, BiCodec and FocalCodec@50 at 0.65, and the low-frame-rate DualCodec at 0.75–0.93. Each scores UTMOS around 4.0–4.2, about the same as the LibriSpeech reference (4.09), with dWER of 2–3%. In 2022–23, EnCodec and DAC scored UTMOS 1.3–1.6 at similar bitrates. Below about 500 bps the picture is less clear-cut. FocalCodec still gets 3.3% dWER at 330 bps. At 160 bps its naturalness scores stay high, but English dWER rises to 8% and multilingual dWER to 54%. The headline sub-100 bps "codecs" only work because the decoder is given the transcript. The usable floor for a stand-alone codec that handles any speaker and any language is therefore about **300–650 bps of raw token rate**. That is still 3–6× above the roughly 100 b/s information-theoretic estimate, and almost nobody entropy-codes their tokens yet. Three caveats shape everything else in this report. First, most results are self-reported, and the most popular no-reference metric (UTMOS) rewards decoders that produce fluent but wrong speech. Second, production systems are years behind research. Deployed neural audio is mostly decoder-side enhancement, packet-loss concealment and redundancy in the 100–600 MFLOPS range. The first standardized end-to-end neural voice codec (3GPP ULBC, 1.0/1.4 kbps) is targeted for early 2027. Third, for music, neural codecs clearly win below about 16 kbps. No rigorous test shows any neural codec beating AAC or Opus at the 32–64 kbps rates where music becomes transparent.

## Single-codebook codecs reach reference naturalness at 650 bps

The defining shift of 2024–2026 is the move from multi-codebook RVQ codecs at 50–86 Hz to codecs that use **one large codebook (8k–65k entries) at 25–80 Hz**, or a few codebooks at 12.5 Hz. In the FocalCodec paper, every codec was run through a single LibriSpeech harness. EnCodec at 1.5 kbps scored UTMOS 1.58 and DAC at 1 kbps scored 1.29. BigCodec at 1.04 kbps scored **4.11 with dWER 2.55%**. FocalCodec@50 at 0.65 kbps scored **4.05 with dWER 2.18%**, the best intelligibility in the table ([FocalCodec](https://arxiv.org/html/2502.04465)). The Spark-TTS paper used a separate harness and got the same ordering: X-Codec2 (50 Hz × 16-bit FSQ = 0.8 kbps) scored UTMOS 4.13, and BiCodec (0.65 kbps) scored 4.18 with PESQ-WB 2.51. For comparison, a 6 kbps DAC with 12 codebooks reaches PESQ-WB 4.01 and SIM 0.98, which is effectively transparent ([Spark-TTS](https://arxiv.org/html/2503.01710)). DualCodec ran its own MUSHRA test, where its 12.5 Hz model scored **88.2 at 0.93 kbps**, against 72.8 for Mimi and 36.2 for EnCodec at 1.5 kbps ([DualCodec](https://arxiv.org/html/2505.13000)).

DualCodec also ran an ablation showing that much of the gain is not exotic. A reproduced DAC run at 12.5 Hz and 0.75 kbps reaches UTMOS 3.87 and MUSHRA 81.8. Most of the improvement comes from **a low frame rate plus a large first codebook** ([DualCodec](https://arxiv.org/html/2505.13000)). Above about 1.5 kbps, 2025–26 research on speech quality has largely stopped, because the problem is saturated there.

| Rate (raw) | Leading codecs | Typical clean-English result | Caveat |
|---|---|---|---|
| ~0.65–1.04 kbps | FocalCodec@50, BiCodec, X-Codec2, DualCodec, BigCodec | UTMOS 4.0–4.2, dWER 2–3% | Independent harnesses agree on ranking |
| ~0.8 kbps streaming | ZipCodec (6.25 Hz, 160 ms), FocalCodec-Stream (80 ms) | UTMOS 3.85–3.89, dWER 2.8–3.7% | Same group built baselines |
| ~0.4–0.5 kbps | TAAE-400, P2PSynCodec | PESQ 2.96 / UTMOS 3.95 | TAAE has 950M params; P2PSyn reports no WER |
| ~330 bps | FocalCodec@25 | dWER 3.3%, MLS dWER 19.8% | Multilingual degradation begins |
| ~160–250 bps | FocalCodec@12.5, FMelCodec | dWER ~8% EN, 54% MLS | Naturalness metrics stay high |
| <100 bps | TaDiCodec (87.5 bps) | TTS WER 2.3% | Decoder needs the transcript |

Streaming results near 0.8 kbps have also improved. **ZipCodec** (September 2026) runs at 6.25 Hz and quantizes 64 dimensions with spherical scalar quantization, giving 128 bits per frame, with a theoretical latency of 160 ms. It reports UTMOS 3.89, dWER 2.83% and Sim 98.0 on LibriSpeech. On MLS its dWER is 15.5%, against 26.1% for Mimi at 0.83 kbps ([ZipCodec](https://arxiv.org/html/2609.11642v1)). FocalCodec-Stream reaches dWER 4.63% at 0.55 kbps with 80 ms latency ([FocalCodec-Stream](https://arxiv.org/html/2509.16195)). Both papers come from the same group (Della Libera, Subakan, Ravanelli), so the baselines are internally consistent but not independent. P2PSynCodec transmits one 1024-entry VQ at 50 Hz (0.5 kbps). The decoder predicts three further "pseudo-VQ" stages that cost no bits. It matches a retrained BigCodec on UTMOS (3.95 vs 3.94) with 23M parameters instead of 159M, but it is lower on STOI and ViSQOL and reports no WER ([P2PSynCodec](https://arxiv.org/html/2606.05876)).

### Below 500 bps, naturalness and intelligibility diverge

FocalCodec's three operating points show the central problem below 500 bps. Going from 650 to 330 to 160 bps, **UTMOS rises (4.05 → 4.14 → 4.22) while dWER rises (2.18 → 3.30 → 7.94%)**. Multilingual dWER climbs from 12.6% to 54.2% ([FocalCodec](https://arxiv.org/html/2502.04465)). Stable Codec (TAAE) shows the same pattern at 0.7 kbps. It has the highest UTMOS in the harness (4.32, above the 4.09 reference) but worse dWER (4.97%) than BigCodec and **57% dWER on MLS**. A score above the reference means the decoder is producing speech cleaner than the input, which is not the same as faithful speech. The likely mechanism is that heavily semantic, English-trained codecs work like a phonetic recognizer feeding a vocoder. On languages they were not trained on, they produce plausible but wrong phones. The October 2026 AfriSpeech study supports this. FocalCodec at 12.5 Hz reached dialog WER 57% (uncompressed reference 17.3%), and SemantiCodec at 0.31 kbps reached 76%. DAC at 16 kbps stayed at 28% ([AfriSpeech robustness](https://arxiv.org/html/2610.00154)).

The sub-100 bps results come from a different category of system. **TaDiCodec** reaches 87.5 bps with one codebook at 6.25 Hz, but its diffusion decoder is conditioned on text ([TaDiCodec](https://arxiv.org/pdf/2508.16790)). It is a tokenizer for TTS and speech LMs, and its rate is not comparable to a communications codec that has to carry content without side information. Several 2026 papers fill in the 250–500 bps range, all self-reported: FMelCodec at 250 bps (a flow-matching mel refiner in front of HiFi-GAN) ([FMelCodec](https://arxiv.org/abs/2605.25669)), LSCodec at 0.25/0.45 kbps with speaker decoupling ([LSCodec](https://www.isca-archive.org/interspeech_2025/guo25_interspeech.pdf)), and AugCodec with separate semantic, speaker and prosody tokens ([AugCodec](https://arxiv.org/abs/2606.21893)). No independent listening test below 300 bps was found.

### Today's codecs sit 3–6× above the information bound

Cross-linguistic studies find spoken language carries about **39 bits/s** of linguistic information, even though syllable rates vary widely between languages ([CNRS / Coupé et al.](https://www.cnrs.fr/en/press/similar-information-rates-across-languages-despite-divergent-speech-rates)). Van Kuyk, Kleijn and Hendriks estimate an upper bound of about **100 b/s** for the acoustic information that actually reaches listeners ([TU Delft](https://repository.tudelft.nl/file/File_94bfb808-583c-4ef2-84b9-91acc8394182)). Stand-alone codecs with good intelligibility run at 300–650 bps before entropy coding, which is 3–6× above that bound. They also have to carry speaker identity and prosody, and no accepted bound exists for either. FocalCodec@12.5 at 160 bps is only about 1.6× above the 100 b/s figure, and its intelligibility collapse suggests it is already hitting the content limit without strong language priors. A plausible near-term target is 150–250 bps at today's 330 bps quality, reached through entropy coding plus speaker and prosody factorization. That is an extrapolation, not a reported result.

## Implicit quantizers, low frame rates and big transformers define the 2026 recipe

### FSQ and binary quantizers have displaced explicit codebooks

The codebook-collapse problem that needed EMA updates, dead-code restarts and commitment losses in EnCodec has been solved several ways. One is DAC's low-dimensional factorized, L2-normalized lookup ([DAC](https://arxiv.org/abs/2306.06546)). Another is SimVQ's reparameterization, which reports 94.7–100% utilization regardless of codebook size ([SimVQ](https://arxiv.org/html/2411.02038v1)). The most common choice for new codecs is an **implicit codebook**: FSQ (X-Codec2's 4^8 = 65,536 codes; TAAE; NanoCodec; Kanade) or binary spherical quantization (FocalCodec, ZipCodec). Measured utilization is now about 0.95 or higher across all of these. TAAE reports normalized entropy of 0.97 for FSQ, 0.87–0.99 for DAC's RVQ levels and 0.91–0.92 for Mimi ([TAAE](https://arxiv.org/html/2411.19842)). Utilization therefore no longer separates quantizers. What does separate them:

- **Channel robustness.** FSQ codes degrade gracefully under bit flips because neighboring levels sound similar, while RVQ indices are arbitrary ([FSQ robustness](https://arxiv.org/pdf/2509.09550)).
- **Post-hoc bitrate flexibility.** TAAE can change FSQ levels or switch to residual FSQ after training, without retraining.
- **LM-friendliness.** A large implicit vocabulary has to be factorized before an LM can predict it. At 4.17 Hz with about 288 bits per frame, a factorized-FSQ NAR head gives TTS WER of 1.83%, against 10.17% with a linear head ([frame-rate study](https://arxiv.org/html/2606.12199)).

RVQ survives in two places: where quantizer dropout provides bitrate scalability, and in the DAC/EnCodec/Mimi ecosystems. It also dominated the compute-capped LRAC challenge (see below).

### Frame rates fell to 12.5–25 Hz, with sub-10 Hz research

Typical frame rates moved from 75–86 Hz (EnCodec, DAC) to 12.5–25 Hz (Mimi, TAAE, DualCodec, Qwen3-TTS-Tokenizer). 2026 research tokenizers go down to 2–8 Hz or use variable rates (FlexiCodec, DyCAST, Locodec). For pure reconstruction, a lower frame rate with more bits per frame is neutral or slightly better: DualCodec's 12.5 Hz model scores a slightly higher UTMOS than its 25 Hz model. For downstream LMs there is a tradeoff. In DualCodec's controlled comparison, VALL-E TTS WER was **3.40% at 25 Hz vs 4.40% at 12.5 Hz**, while real-time factor roughly halved ([DualCodec](https://arxiv.org/html/2505.13000v1)). A June 2026 matched study held information rate at 600 bps and used a frozen Qwen3-4B. ASR WER was U-shaped, best between 12.5 and 4.17 Hz. TTS WER degraded sharply once speech tokens became fewer than text tokens (about 3.3 tokens/s on LibriSpeech). Spoken QA peaked at 4–6 Hz ([frame-rate study](https://arxiv.org/html/2606.12199)). The practical reading: fixed 12.5–25 Hz is the codec standard, and content-adaptive rates close to the text rate are the direction for spoken-language models.

### Scale and transformers lead the high-resource frontier

TAAE (Stability AI) showed that a 950M-parameter patched transformer autoencoder with FSQ beats conv/RVQ codecs at 400–700 bps. Its own scale ablation, however, gained only 0.08 PESQ going from 240M to 950M ([TAAE](https://arxiv.org/html/2411.19842)). In 2026, MOSS-Audio-Tokenizer (1.6B parameters, about 3M hours of general audio, purely transformer, no pretrained encoders) and TAC (ICML 2026) both report predictable gains from scale ([MOSS-Audio-Tokenizer](https://arxiv.org/abs/2602.10934); [TAC](https://proceedings.mlr.press/v306/gong26c.html)). Lightweight codecs still use convolutional SEANet/DAC-style stacks with Snake activations, often with LSTM or GRU layers (BigCodec). For cheap decoding from low frame rates, Vocos-style iSTFT heads are standard, at the cost of slight "phasiness" compared with time-domain BigVGAN-style decoders ([IWAENC 2026](https://audiolabs-erlangen.de/content/resources/NLUI/00_2026-IWAENC-vocoder/main.pdf)). No codec paper found evidence that Mamba/SSM backbones help.

### Semantic supervision helps generation; scale is the counter-argument

For codecs that feed speech LMs, semantic supervision clearly works. DualCodec's ablation measured Seed-TTS-Eval WER of first-layer tokens:

| DualCodec variant | WER EN / ZH (%) |
|---|---|
| DAC baseline | 55.4 / 46.4 |
| + SSL distillation | 28.4 / 26.4 |
| + parallel w2v-BERT-2.0 semantic branch | 5.59 / 6.52 |
| + 16,384-entry first codebook | **2.98 / 2.91** |

Reconstruction MUSHRA also improved, from 68.8 to 79.5 at 25 Hz ([DualCodec](https://arxiv.org/html/2505.13000v1)). Two lessons follow. Distilling semantics into a shared first RVQ level costs reconstruction quality. Parallel branches (Mimi's split RVQ, X-Codec's feature fusion, DualCodec's dual encoders) avoid that cost. MOSS-Audio-Tokenizer argues the opposite: that end-to-end training at scale makes pretrained encoders and distillation unnecessary ([MOSS](https://arxiv.org/abs/2602.10934)). This has not been independently confirmed.

### Generative decoders shift the naturalness frontier, not the fidelity frontier

GAN decoders with multi-scale STFT and multi-period discriminators plus feature matching remain the default for real-time codecs. Rate–distortion–perception theory explains why generative decoders appear at very low rates. Perfect perceptual quality costs at most 2× MSE at a fixed rate, and may require stochastic decoding ([Blau & Michaeli](https://arxiv.org/abs/1901.07821)). Mimi applies this in practice: it trains with adversarial and feature-matching losses only ([Moshi](https://arxiv.org/abs/2410.00037)). TAAE gets **+0.27 PESQ and +0.04 STOI** from fine-tuning with a WavLM perceptual loss, with SI-SDR nearly unchanged ([TAAE](https://arxiv.org/html/2411.19842)).

Diffusion and flow decoders, the extreme form of this approach, have a mixed record:

- **SemantiCodec** (diffusion, 507M parameters) scored only UTMOS 2.91 and dWER 8.97% at 0.65 kbps. Its outputs are misaligned in time, so SI-SDR cannot be computed ([FocalCodec](https://arxiv.org/html/2502.04465); [TAAE](https://arxiv.org/html/2411.19842)).
- **FlowDec** is only "on par" with GAN-based DAC at 4.5–7.5 kbps ([FlowDec](https://arxiv.org/html/2503.01485)).
- **2026 low-frame-rate tokenizers** (FlexiCodec, SiTok at 12.5 Hz/200 bps, Locodec at 8 Hz continuous) use flow or diffusion heads because their goal is generation ([SiTok](https://arxiv.org/abs/2602.06602); [Locodec](https://arxiv.org/abs/2607.29363)).

Overall, generative decoding moves the naturalness frontier down by roughly 2–4× in bitrate. The intelligibility and speaker-fidelity frontier moves much less. The costs are hallucination, speaker drift, lost waveform alignment, 5–10× more parameters, and sampling latency.

### Entropy coding is the most neglected free lunch

Almost every paper reports raw rate, meaning frame rate × log₂(codebook size). For well-utilized codecs, static Huffman coding saves little. TAAE's table shows −16% for EnCodec but only −4 to −9% for DAC, Mimi and TAAE itself ([TAAE](https://arxiv.org/html/2411.19842)). Context modeling saves more: EnCodec's small transformer prior cut 6 kbps to 4.1 kbps, and EnCodec reported savings of up to 40% ([EnCodec](https://arxiv.org/pdf/2210.13438)). ECC (June 2026) is one of the few speech codecs with explicit entropy models (hyperprior plus channel context). It claims BD-rate savings of 39.9% on ViSQOL and 76.3% on PESQ ([ECC](https://arxiv.org/abs/2606.11631)). No 2025–26 paper was found that arithmetic-codes modern low-rate tokens (Mimi, X-Codec2, FocalCodec) under a strong LM prior. Since speech LMs already learn exactly that conditional distribution, the real bitrate of today's best codecs is probably meaningfully lower than reported.

## Benchmarks disagree and UTMOS rewards hallucination

### The LRAC challenge is the strongest human-rated evidence

The 2025 Low-Resource Audio Codec challenge, run by Cisco with CMU, UIUC and HUJI, collected about **186,720 crowdsourced ratings** (MUSHRA, DCR/DMOS and the Diagnostic Rhyme Test) on 17 codecs. It had two bitrate modes, ≤1 kbps and ≤6 kbps, and tight compute limits. In the transparency track, receive-side compute was capped at 300 MFLOP/s, total compute at 700 MFLOP/s, and latency at 30 ms ([LRAC description](https://arxiv.org/abs/2510.23312); [results](https://crowdsourcing.cisco.com/lrac-challenge/2025/results)). ByteDance's IRIS won Track 1, and NJU's PR-Vocodec won the enhancement track. Note that the challenge's "NanoCodec" from CAS/ByteDance is a different model from NVIDIA's NanoCodec.

The organizers' findings matter more than the rankings ([LRAC overview](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)):

- **At 6 kbps**, compute-capped codecs were comparable to state-of-the-art high-resource codecs.
- **At 1 kbps**, high-resource codecs "significantly outperform" them. These codecs cost one to two orders of magnitude more compute, and DualCodec has about 4 s of latency without its offline w2v-BERT front end.
- **Robustness, not clean-speech quality, decided the ranking.** The team with the best clean-speech score fell to 4th–5th under noise and reverberation.
- **Enhancement codecs reduced intelligibility** even on clean input, with sustained consonants worst hit.
- **The winning designs** used STFT/mel front ends, ConvNeXt-type blocks with GRU or LSTM layers, factorized RVQ and GAN training. None used transformer-only, autoregressive or diffusion decoders.

### Which metrics track listeners

| Metric | Lanzendörfer, speech-only | Lechler, DNN codecs | Notes |
|---|---|---|---|
| SCOREQ (reference) | **0.937** | **−0.79** (distance) | Also best in LRAC (0.87) |
| PESQ | 0.886 | 0.58 | Underestimates DNN codecs in absolute terms |
| ViSQOL | 0.850 (speech mode) | 0.30 | |
| DNSMOS | 0.683 | 0.29 | −0.07 with background audio |
| NISQA | 0.563 | 0.18 | |

Three studies agree that **reference-based SCOREQ is the best objective proxy for MUSHRA** on codec speech. PESQ ranks homogeneous sets of GAN codecs well, but it underestimates neural codecs against DSP codecs and is unreliable for generative ones ([Lanzendörfer & Grötschla](https://arxiv.org/abs/2511.19734); [Lechler et al.](https://www.isca-archive.org/interspeech_2025/lechler25_interspeech.pdf); [LRAC overview](https://crowdsourcing.cisco.com/lrac-workshop/2026/assets/files/2026-LRAC-Slides-Challenge-Overview.pdf)). No-reference MOS predictors saturate at high quality and correlate poorly with listeners. Every one of these studies used only 4–6 codecs and none reports confidence intervals. The rank order SCOREQ ≈ PESQ > ViSQOL > NISQA/DNSMOS is robust; the individual coefficients are not.

The failure cases are concrete:

- In Kanade's table, Stable Codec has the top UTMOS (4.31, above the 4.07 ground truth) and **three times the reference WER** (5.7% vs 1.9%) ([Kanade](https://arxiv.org/html/2602.00594)).
- In the AfriSpeech study, WavTokenizer ranked 1st of 28 on NISQA with WER 68%. FocalCodec@12.5 ranked 1st on UTMOS but did poorly on speaker EER. UTMOS correlated with dialog WER at only ρ = −0.26. Swapping the speaker-verification backend reversed the speaker-similarity rankings ([AfriSpeech](https://arxiv.org/html/2610.00154)).
- Speaker similarity is not portable across papers. BigCodec scores 98.5, 0.84 and 0.62 under three harnesses ([FocalCodec](https://arxiv.org/html/2502.04465); [Spark-TTS](https://arxiv.org/html/2503.01710); [VERSA](https://arxiv.org/pdf/2412.17667)). X-Codec2 scores 0.82 in Qwen's table and 0.98 in Kanade's ([Qwen3-TTS](https://arxiv.org/html/2601.15621)).

What does replicate across independent harnesses is the overall ordering. The single-codebook class (BigCodec, X-Codec2, BiCodec, FocalCodec) beats EnCodec, DAC, SpeechTokenizer and WavTokenizer near 1 kbps, and Mimi underperforms per bit at matched sub-kbps rates. Every 2026 frontier claim (ZipCodec, P2PSynCodec, FMelCodec, ECC, VoCodec, PACodec) is self-reported, often against retrained rather than official baselines.

### Benchmark suites measure different things

Codec-SUPERB resynthesizes audio and runs fixed downstream models on it. In its SLT 2024 edition, STOI correlated best with downstream utility (−0.81 with ASR WER, −0.89 with speaker EER). SemantiCodec's diffusion time-shifts broke the signal metrics ([Codec-SUPERB SLT](https://arxiv.org/html/2409.14085v1)). DASB (TMLR 2026) instead trains downstream models on the tokens. It found semantic tokens beat compression tokens on most tasks, and all discrete tokens still trail continuous SSL features substantially ([DASB](https://arxiv.org/html/2406.14294v1)). The two protocols answer different questions, transmission fidelity versus usefulness as a representation, and a codec can win one and lose the other. VERSA (65 metrics) is the common metric toolkit ([VERSA](https://awesomepapers.io/speech-audio/papers/shi2024versa)). A defensible protocol combines:

- SCOREQ and PESQ-WB for quality;
- WER from two ASR models;
- speaker similarity from two backends;
- F0 error for prosody;
- UTMOS only as a secondary signal, flagging any score above the reference;
- a small MUSHRA-1S confirmation ([MUSHRA-1S](https://arxiv.org/pdf/2509.19219)).

## Production ships neural add-ons, not neural codecs

### Opus shows the deployable pattern

Opus is the most widely deployed codec with neural components, and Xiph's approach has been to keep the RFC 6716 bitstream and add ML only where old decoders can safely ignore it. Opus 1.5 added Deep PLC, the LACE/NoLACE decoder-side enhancers and experimental DRED. **Opus 1.6 (December 15, 2025)** added an ML wideband-to-fullband bandwidth extension, a smaller DRED model (about 600 kB, down from 1800 kB) that is incompatible with 1.5's, and "Opus HD" high-resolution extensions. In Xiph's own tests, NoLACE plus BWE at 9 kb/s matches Opus 1.4 fullband at 18 kb/s and "closes the gap with EVS" ([Opus 1.6](https://opus-codec.org/demo/opus-1.6/)). DRED sends RDO-VAE latents, one per 40 ms, at about **1/50 of the Opus bitrate**, so a single packet can carry a second or more of redundancy for burst loss. Its vocoder is limited to roughly 5 ms of look-ahead. The IETF draft is at revision 07 (August 2026), with no RFC yet ([draft-ietf-mlcodec-opus-dred-07](https://www.ietf.org/ietf-ftp/internet-drafts/draft-ietf-mlcodec-opus-dred-07.html)). At a typical 24–48 kb/s Opus rate, the DRED stream is effectively a deployed **sub-kbps neural speech codec used only for redundancy**. These features have tight compute budgets ([LACE](https://jmvalin.ca/slides/LACE_waspaa.pdf); [NoLACE](https://arxiv.org/pdf/2309.14521); [FARGAN](https://arxiv.org/pdf/2307.06610)):

| Opus feature | Compute | Parameters |
|---|---|---|
| LACE enhancer | ~100 MFLOPS | 300K |
| NoLACE enhancer | 400–620 MFLOPS | — |
| FARGAN vocoder (DRED and PLC synthesis) | ~600 MFLOPS | — |

### Company deployments are low-bandwidth fallbacks

| System | What it is | Operating point | Vendor claim |
|---|---|---|---|
| Microsoft Satin | Decoder-side DNN bandwidth extension | 6 kb/s+ | +1.7 MOS over SILK at 6 kb/s ([Microsoft](https://techcommunity.microsoft.com/blog/microsoftteamsblog/satin-microsoft%E2%80%99s-latest-ai-powered-audio-codec-for-real-time-communications/2141382)) |
| Cisco Webex AI Codec | Neural codec + ~1 kb/s redundancy | ~6 kb/s, triggered at >30% loss | Intelligible at 80–90% loss ([Webex](https://blog.webex.com/collaboration/announcing-the-availability-of-industry-first-webex-ai-codec-transforming-your-audio-experiences/); [heise](https://heise.de/-9848886)) |
| Google Lyra (Meet) | SoundStream-based end-to-end codec | ≥3 kb/s, low-bandwidth mode | Niche; limited by compute and device support ([Red5](https://www.red5.net/blog/what-does-the-new-google-lyra-audio-codec-mean-for-real-time-video-streaming/)) |
| Meta MLow | Low-compute CELP (not neural) | 6 kb/s WB | POLQA ~3.9 vs 1.89 for Opus at 6 kb/s ([Meta](https://engineering.fb.com/2024/06/13/web/mlow-metas-low-bitrate-audio-codec/)) |

Every comparison in this table comes from the vendor. The pattern is consistent: large wins at about 6 kb/s, parity by about 20–25 kb/s (MLow and Opus are equal at about 25 kb/s), and no major VoIP platform has publicly replaced its main wideband codec with an end-to-end neural one. Zoom documents only Opus and SILK ([Zoom](https://library.zoom.com/admin-corner/network-management/quality-of-service-and-network-best-practices-explainer/how-zoom-audio-and-video-codecs-affect-bandwidth-usage)). No primary source was found for Apple, Discord or Signal.

### 3GPP ULBC will be the first standardized neural voice codec

3GPP SA4's Rel-20 **Ultra-Low Bitrate speech Codec (ULBC)** targets **1.0 and 1.4 kb/s** for IMS voice over GEO satellite via NB-IoT NTN. Candidates are expected to be AI-based, "a first for 3GPP voice codec standardization," and the first standards are targeted for **early 2027**. Algorithmic delay, bandwidth and model complexity and memory limits are still under discussion ([Samsung Research](https://research.samsung.com/blog/Voice-Over-NB-IoT-NTN-In-3GPP-Release-20); [3GPP SA4 work items](https://www.3gpp.org/dynareport?code=TSG-WG--S4--wis.htm)). IVAS, 3GPP's current immersive voice codec, is classical. It was demonstrated on commercial devices at MWC 2026, but wide deployment is "still on the horizon" ([3GPP IVAS](https://www.3gpp.org/technologies/ivas-06-2026); [Nokia](https://www.nokia.com/multimedia/audio/immersive-voice/faq/)). MPEG has no neural audio coding call; its nearest work is Audio Coding for Machines ([MPEG](https://www.mpeg.org/emerging-technologies/)).

### Latency and compute separate research codecs from telephony

Deployed real-time codecs work with 20 ms frames, strictly causal neural components, and decoder add-ons of 100–600 MFLOPS. Research tokenizers run at much longer time scales:

- **Mimi:** 12.5 Hz, so each frame is 80 ms ([Moshi](https://arxiv.org/abs/2410.00037)).
- **Qwen3-TTS-Tokenizer-12Hz:** 320 ms packets.
- **ZipCodec:** 160 ms theoretical latency.
- **TAAE:** 950M parameters.

This explains the LRAC result: at a 700 MFLOP/s budget and 30 ms latency, the field can match high-resource codecs at 6 kbps but not at 1 kbps. The sub-kbps quality frontier currently exists only where latency and compute are relaxed. That makes GEO satellite (where round-trip time dwarfs codec delay), messaging and LLM tokenization the natural first homes for it, not terrestrial duplex calls.

## Music codecs win below 16 kbps but not at transparency

For general audio and music, neural codecs clearly beat classical codecs at low rates. In EnCodec's 2022 48 kHz stereo music MUSHRA, **EnCodec at 6 kbps scored 82.9, equal to MP3 at 64 kbps (82.7) and far above Opus at 6 kbps (17.7)** ([EnCodec](https://arxiv.org/pdf/2210.13438)). DAC (44.1 kHz, up to 8 kbps) remains the baseline that nearly every 2025–26 paper compares against, though its authors concede it falls short of the reference even at 8 kbps ([DAC](https://arxiv.org/pdf/2306.06546)). The 2025–26 leaders all target low-rate tokenization:

- **SpectroStream** (Google DeepMind): 48 kHz stereo, causal, 80 ms latency, 61M parameters. Against DAC on ViSQOL it scores **3.21 vs 1.47 at 2.7 kbps per channel** and 4.00 vs 3.33 at 8 kbps. Raters preferred it 76% of the time at the lowest rate, but there was no Opus or AAC anchor ([SpectroStream](https://arxiv.org/html/2508.05207v1)).
- **FlowDec** (Meta/Hamburg): matches DAC on MUSHRA at 4–7.5 kbps and beats it on FAD ([FlowDec](https://arxiv.org/pdf/2503.01485)).
- **Continuous VAEs for diffusion generation** now match or beat 86 Hz DAC at 25–37 Hz. Qwen-Audio-3.0's 25 Hz stereo VAE scores ViSQOL 4.30 vs DAC's 4.20 on Song Describer ([Qwen-Audio-3.0](https://arxiv.org/pdf/2607.27011)). Adobe's GenAE matches DAC at about 15% of DAC's latent rate ([GenAE](https://arxiv.org/html/2602.15749v1)).

These continuous latents are not bitstreams. No paper was found that entropy-codes a music VAE.

At the rates where music becomes transparent, the evidence for neural codecs is absent. **No rigorous MUSHRA, BS.1116 or ABX test shows any neural codec beating AAC-LC/HE-AAC or Opus at 32–64 kbps stereo.** At 24 kbps, Gull-L (92.5) and Opus (92.3) both sit at the MUSHRA ceiling (reference 92.8), and ViSQOL favors Opus at ≥20 kbps ([Gull](https://arxiv.org/html/2404.04947v2)). EnCodec at 24 kbps plateaued about 7 points below the reference. TQCodec (Tencent Music, 2026) is the only paper found that targets 32–128 kbps. Its 64 kbps model was preferred no more often than Ogg Vorbis at 96 kbps (39.1% vs 39.4%), and the test had no AAC or Opus anchor and no listener count ([TQCodec](https://arxiv.org/html/2603.01592)). TQCodec's high-band error stays flat from 32 to 128 kbps, which suggests GAN-style decoders synthesize plausible high frequencies rather than reproduce them. That works for MOS but blocks transparency.

Stereo remains crude. TQCodec found a joint two-channel network "performed poorly" and codes stereo as dual mono. SpectroStream reports that fusing channels too late loses phase coherence. Stable Audio Open's 21.5 Hz VAE degrades inter-channel phase coherence (0.57 vs about 0.70 for others) ([Qwen-Audio-3.0](https://arxiv.org/pdf/2607.27011)). In Fraunhofer's ODAQ test, DAC struggled on tonal and panned items ([ODAQ MUSHRA](https://arxiv.org/html/2509.18823)). FAD computed on codec embeddings correlates only weakly with listeners (Pearson 0.38 for EnCodec embeddings), so objective "near-transparent" claims need listening tests behind them.

Universal codecs pay a speech penalty only when capacity is scarce. At 75 tokens/s, a unified WavTokenizer lost 5 speech MUSHRA points and 19 music points compared with its domain-specific siblings. UniCodec's partitioned codebook with a domain mixture-of-experts beat both, scoring 90.7 on speech and 77.8 on music ([UniCodec](https://arxiv.org/html/2502.20067v1)). Speech-only codecs fail badly on music: BigCodec's music mel distance is 1.38 vs DAC's 0.36.

## Mimi, Qwen and DAC lead the usable open weights

Licenses vary as much as quality does, and they often decide which codec is usable.

| Model | License | Rate / structure | Streaming | Standing |
|---|---|---|---|---|
| [Mimi](https://hf.co/kyutai/mimi) | CC-BY-4.0 | 24 kHz, 12.5 Hz split RVQ, 1.1 kbps @ 8 cb | Causal | Most adopted (22.6M downloads); weaker per bit than single-codebook codecs |
| [Qwen3-TTS-Tokenizer-12Hz](https://hf.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) | Apache-2.0 | 12.5 Hz, 16×2048 (~2.2 kbps) | Fully causal, 320 ms packets | Vendor table best: PESQ-WB 3.21, SIM 0.95 ([report](https://arxiv.org/html/2601.15621)) |
| [NVIDIA NanoCodec](https://hf.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps) | NVIDIA Open Model License | 22 kHz FSQ; 0.6–1.89 kbps | Causal decoder | Used by Magpie-TTS |
| [XCodec2](https://hf.co/HKUSTAudio/xcodec2) | **CC-BY-NC-4.0** | 16 kHz, 50 Hz, 1×65,536 FSQ, 0.8 kbps | No | Pareto-frontier reconstruction; Llasa backbone |
| [NeuCodec](https://hf.co/neuphonic/neucodec) | Apache-2.0 (gated) | XCodec2-style, 0.8 kbps | Not stated | Permissive XCodec2 alternative; no independent benchmark |
| [BiCodec](https://hf.co/SparkAudio/Spark-TTS-0.5B) | **CC-BY-NC-SA-4.0** | 50 Hz semantic + global speaker tokens, ~0.65 kbps | No | UTMOS 4.18, WER 2.5% (Kanade table) |
| [Kanade](https://hf.co/frothywater/kanade-25hz) | MIT | 12.5/25 Hz single-token FSQ | Not stated | WER 2.4%, UTMOS 4.16 (own table) ([paper](https://arxiv.org/html/2602.00594)) |
| [DualCodec](https://hf.co/amphion/dualcodec) | Not shown on card | 12.5/25 Hz, ~0.75–0.93 kbps | **No** (~4 s latency) | LRAC high-resource reference that beat capped codecs at 1 kbps |
| [Stable Codec](https://hf.co/stabilityai/stable-codec-speech-16k) | Custom, gated | 16 kHz, 25 Hz RFSQ | — | Hallucination case: top UTMOS, high WER |
| [DAC](https://hf.co/descript/descript-audio-codec) | MIT | 16/24/44.1 kHz RVQ, to ~8 kbps | No | Best MUSHRA and ASR/ASV preservation in independent tests |
| [SNAC](https://hf.co/hubertsiuzdak/snac_24khz) | MIT | Multi-scale RVQ, ~1 kbps | — | Orpheus TTS backbone |
| [WavTokenizer](https://hf.co/novateur/WavTokenizer-large-speech-75token) | MIT | 40/75 Hz single VQ | No | Poor independent intelligibility; avoid for content |

Licenses, parameter counts and download figures come from the linked Hugging Face cards as of October 30, 2026. The standing column draws on independent comparisons in [Kanade](https://arxiv.org/html/2602.00594), [AfriSpeech](https://arxiv.org/html/2610.00154) and [Lanzendörfer & Grötschla](https://arxiv.org/abs/2511.19734).

Recommendations by use case:

- **Low-bitrate speech compression with a permissive license:** DAC at 1.5–6 kbps, or Mimi at 1.1 kbps if streaming is needed.
- **Best quality at ≤1 kbps, offline:** DualCodec, BigCodec or XCodec2/NeuCodec. Check the licenses, and note DualCodec's latency. FocalCodec's weights are on [GitHub](https://github.com/lucadellalib/focalcodec) rather than Hugging Face.
- **Tokenizers for speech LMs:**
  - Delay/depth-transformer designs: Qwen3-TTS-Tokenizer-12Hz or Mimi.
  - Plain next-token LMs: NeuCodec, Kanade or XCodec2.
- **General audio and music:** DAC 44.1 kHz and EnCodec remain the safe open baselines.

Several large 2026 releases have no independent evaluation yet: MiMo-Audio-Tokenizer (1.3B, MIT, updated October 28, 2026), the Higgs Audio v2 tokenizer, and the LRAC winners, whose weights do not appear to be released (the second edition will mandate open source).

## Five problems remain open

**Honest rate accounting** is the first gap. Without entropy-coded rates, side information (text, global speaker tokens) and multilingual intelligibility, sub-500 bps claims cannot be compared. Text-conditioned tokenizers and stand-alone codecs are routinely put in the same tables.

**Hallucination measurement** is the second. No paper quantifies substituted words or phones per minute for generative decoders. The field's most cited naturalness metric goes up as faithfulness goes down.

**The compute–latency gap** is the third. The quality available at 1 kbps needs about 10–100× the compute and 4–100× the latency that telephony allows. ULBC will test whether a constrained neural codec can close that gap by 2027.

**Robustness** is the fourth. LRAC rankings flipped under noise, reverberation and overlapping speakers. Bursty packet loss is barely studied outside Opus DRED and Webex, and English-trained codecs fail on African and other low-resource languages.

**Music transparency** is the fifth. No controlled test of a modern neural codec against Opus 1.5+ or AAC at 32–64 kbps with critical items exists. Stereo and multichannel coding have no neural equivalent of M/S or intensity-stereo tools.

Two smaller questions are also unresolved. No ablation holds everything else fixed and swaps only the quantizer, and no independent test shows whether scale can replace semantic distillation.

## Conclusion

The main change over the past two years is that bitrate is no longer the binding constraint for speech. Faithfulness, compute and evaluation are. At about 650 bps, the field already has codecs that listeners and metrics rate as near-reference on clean English. What it lacks is evidence that they hold up on any language, any speaker and any noise condition, at a compute and latency cost a phone call can afford. The steepest remaining gains are probably not new architectures. More likely candidates are LM-prior entropy coding of existing tokens (plausibly tens of percent for free), factorizing speaker and prosody into slowly updated side channels, and robustness training of the kind that decided LRAC. Together these point toward roughly 150–250 bps stand-alone speech within reach of today's model classes.

The second implication is about evaluation. Self-reported UTMOS tables are the field's default currency, and it is not trustworthy below 1 kbps. Any claim about the frontier should come with SCOREQ, WER from two ASR models on multilingual data, speaker similarity from two backends, and a crowdsourced MUSHRA-1S check. Several 2026 results may not survive that standard. For music, the two lines of work have split. Neural models now set the state of the art for generative latents and sub-16 kbps tokens, while classical perceptual codecs still own transparency, so a "neural replaces AAC" result would need a test that has not yet been run.
