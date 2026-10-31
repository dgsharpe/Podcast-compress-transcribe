# mimi-android

The Android side of the `.mimi` codec (FORMAT.md). It builds with AGP 9.4.1 / Gradle 9.8.2, the same as
pocket-casts-android.

| module | what it is |
|---|---|
| `:mimi` | **Media3 extension** (AAR): `MimiExtractor` (`.mimi` → one 1.44 s block per sample, exact seeks), `MimiDecoder` + `LibmimiAudioRenderer`, `MimiRenderersFactory`, `MimiExtractorsFactory`, plus `MimiFileDecoder` (random-access PCM for non-Media3 use). Native: `libmimi_jni` = `phone/libmimi` + JNI, on LiteRT 2.3.2 (NPU transformer via QNN 2.50 / Hexagon V81, GPU conv decoder; falls back to CPU+GPU, then CPU). |
| `:demo` | Test player: open a `.mimi` file or URL; shows the chosen backends and decode timing. |
| `:bridge` | **Mimi Bridge**: lets *any* player play `.mimi`. You "Open with" or "Share" a file to it, and it hands players `content://org.joshjob.mimi.bridge.wav/<name>.wav`, a seekable WAV decoded on read (`StorageManager.openProxyFileDescriptor`). No root, and the player needs no changes. |

## Build

```sh
android/fetch_deps.sh      # libLiteRt.so (link copy), QAIRT QNN libs (range-fetched), Mimi models
android/build.sh           # :mimi:assembleRelease :demo:assembleDebug (in the server's android-build:36 container)
android/build.sh :bridge:assembleDebug
```

## Using `:mimi` in an ExoPlayer app

```java
ExoPlayer player = new ExoPlayer.Builder(context, new MimiRenderersFactory(context))
    .setMediaSourceFactory(new DefaultMediaSourceFactory(context, new MimiExtractorsFactory()))
    .build();
MimiLibrary.preloadAsync(context);   // e.g. in Application.onCreate: decoder setup takes ~2–3 s
```

There are three requirements for the host app:

- `androidResources { noCompress += listOf("tflite", "bin") }`. The library's own setting isn't inherited, and the
  models are copied out of the APK on first use (≈150 MB into `files/mimi/`).
- `packaging { jniLibs { useLegacyPackaging = true } }`. QNN loads the Hexagon skel from a real file in
  `nativeLibraryDir`. ADSP_LIBRARY_PATH points there and `MimiLibrary` sets it.
- 64-bit ARM. `libcdsprpc.so` is requested through `<uses-native-library>` (merged from the AAR).

If the app already subclasses `DefaultRenderersFactory`, call `MimiRenderersFactory.addMimiRenderer(...)` from its
`buildAudioRenderers`. If it has its own `ExtractorsFactory`, wrap it: `new MimiExtractorsFactory(theirs)`.

## Size (2026-10-31)

The assets are 166 MB: two transformer graphs, the conv decoder and the fp16 tables. The QNN runtime libraries
add about 37 MB. The patched Pocket Casts debugProd APK (arm64 only) is 264 MB.

- **fp16 RVQ tables** (`mimi_rvq_f16.bin`, made by `scripts/rvq_fp16.py`) are 34 MB instead of 69 MB. The audio
  stays at 71.7 dB SNR against fp32 tables on the server and 48 dB on the phone, and fp16 decodes faster (39.5 vs 45.9
  ms per block).
- **The NPU graph is precompiled** (`mimi_dec_tx_npu.tflite`, Hexagon V81 / SM8850): it is LiteRT's JIT cache from
  one NPU run, copied in. It loads with only the dispatch library and the QNN runtime. Neither
  `libQnnHtpPrepare.so` (85 MB) nor the compiler plugin ships; apps exclude `libLiteRtCompilerPlugin_Qualcomm.so`.
  On any other SoC, libmimi uses `mimi_dec_tx_fp16.tflite` on the CPU.
- Possible next step: load the models straight out of the APK (`LiteRtCreateModelFromFd` on the uncompressed
  assets) instead of copying them to `files/mimi/`. That would save about 165 MB of storage on the phone.
