package org.joshjob.mimi;

import android.content.Context;
import android.os.Handler;
import androidx.annotation.Nullable;
import androidx.media3.common.C;
import androidx.media3.common.Format;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.common.util.Util;
import androidx.media3.decoder.CryptoConfig;
import androidx.media3.exoplayer.audio.AudioRendererEventListener;
import androidx.media3.exoplayer.audio.AudioSink;
import androidx.media3.exoplayer.audio.DecoderAudioRenderer;

/** Audio renderer for {@code audio/x-mimi}, decoding with the shared native {@link MimiLibrary}. */
@UnstableApi
public class LibmimiAudioRenderer extends DecoderAudioRenderer<MimiDecoder> {
  private static final String TAG = "LibmimiAudioRenderer";
  private static final int NUM_BUFFERS = 8;  // 8 x 1.44 s of audio in flight

  private final Context context;

  public LibmimiAudioRenderer(
      Context context,
      @Nullable Handler eventHandler,
      @Nullable AudioRendererEventListener eventListener,
      AudioSink audioSink) {
    super(eventHandler, eventListener, audioSink);
    this.context = context.getApplicationContext();
  }

  @Override
  public String getName() {
    return TAG;
  }

  @Override
  protected @C.FormatSupport int supportsFormatInternal(Format format) {
    if (!MimiLibrary.MIME_TYPE.equals(format.sampleMimeType) || !MimiLibrary.isAvailable()) {
      return C.FORMAT_UNSUPPORTED_TYPE;
    }
    if (!sinkSupportsFormat(Util.getPcmFormat(C.ENCODING_PCM_16BIT, 1, MimiDecoder.SAMPLE_RATE))) {
      return C.FORMAT_UNSUPPORTED_SUBTYPE;
    }
    return C.FORMAT_HANDLED;
  }

  @Override
  protected MimiDecoder createDecoder(Format format, @Nullable CryptoConfig cryptoConfig)
      throws MimiDecoderException {
    int inputSize =
        format.maxInputSize != Format.NO_VALUE
            ? format.maxInputSize
            : MimiExtractor.SAMPLE_HEADER + MimiExtractor.WINDOW * 44;
    return new MimiDecoder(MimiLibrary.acquire(context), NUM_BUFFERS, inputSize);
  }

  @Override
  protected Format getOutputFormat(MimiDecoder decoder) {
    return Util.getPcmFormat(C.ENCODING_PCM_16BIT, 1, MimiDecoder.SAMPLE_RATE);
  }
}
