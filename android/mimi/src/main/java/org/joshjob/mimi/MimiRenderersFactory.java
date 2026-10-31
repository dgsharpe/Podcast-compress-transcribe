package org.joshjob.mimi;

import android.content.Context;
import android.os.Handler;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.DefaultRenderersFactory;
import androidx.media3.exoplayer.Renderer;
import androidx.media3.exoplayer.audio.AudioRendererEventListener;
import androidx.media3.exoplayer.audio.AudioSink;
import androidx.media3.exoplayer.mediacodec.MediaCodecSelector;
import java.util.ArrayList;

/**
 * {@link DefaultRenderersFactory} with {@link LibmimiAudioRenderer} in front of the default audio
 * renderers. Apps that already subclass DefaultRenderersFactory can call {@link #addMimiRenderer}
 * from their own {@code buildAudioRenderers} instead.
 */
@UnstableApi
public class MimiRenderersFactory extends DefaultRenderersFactory {
  public MimiRenderersFactory(Context context) {
    super(context);
  }

  public static void addMimiRenderer(
      Context context,
      Handler eventHandler,
      AudioRendererEventListener eventListener,
      AudioSink audioSink,
      ArrayList<Renderer> out) {
    out.add(new LibmimiAudioRenderer(context, eventHandler, eventListener, audioSink));
  }

  @Override
  protected void buildAudioRenderers(
      Context context,
      @ExtensionRendererMode int extensionRendererMode,
      MediaCodecSelector mediaCodecSelector,
      boolean enableDecoderFallback,
      AudioSink audioSink,
      Handler eventHandler,
      AudioRendererEventListener eventListener,
      ArrayList<Renderer> out) {
    addMimiRenderer(context, eventHandler, eventListener, audioSink, out);
    super.buildAudioRenderers(
        context, extensionRendererMode, mediaCodecSelector, enableDecoderFallback, audioSink,
        eventHandler, eventListener, out);
  }
}
