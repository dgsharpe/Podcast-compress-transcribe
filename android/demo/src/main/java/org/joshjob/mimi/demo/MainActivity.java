package org.joshjob.mimi.demo;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.EditText;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.annotation.OptIn;
import androidx.media3.common.MediaItem;
import androidx.media3.common.PlaybackException;
import androidx.media3.common.Player;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory;
import androidx.media3.ui.PlayerControlView;
import org.joshjob.mimi.MimiDecoderException;
import org.joshjob.mimi.MimiExtractorsFactory;
import org.joshjob.mimi.MimiLibrary;
import org.joshjob.mimi.MimiRenderersFactory;

/** Minimal test player for the :mimi Media3 extension. */
@OptIn(markerClass = UnstableApi.class)
public class MainActivity extends Activity {
  private static final int OPEN = 1;
  private ExoPlayer player;
  private TextView status;
  private final Handler ui = new Handler(Looper.getMainLooper());
  private String lastError = "";

  @Override
  protected void onCreate(@Nullable Bundle saved) {
    super.onCreate(saved);
    setContentView(R.layout.activity_main);
    MimiLibrary.preloadAsync(this);
    status = findViewById(R.id.status);
    player =
        new ExoPlayer.Builder(this, new MimiRenderersFactory(this))
            .setMediaSourceFactory(new DefaultMediaSourceFactory(this, new MimiExtractorsFactory()))
            .build();
    player.addListener(
        new Player.Listener() {
          @Override
          public void onPlayerError(PlaybackException e) {
            lastError = e.getErrorCodeName() + ": " + e.getMessage()
                + (e.getCause() != null ? " / " + e.getCause() : "");
          }
        });
    ((PlayerControlView) findViewById(R.id.controls)).setPlayer(player);
    EditText url = findViewById(R.id.url);
    findViewById(R.id.play_url).setOnClickListener(v -> play(Uri.parse(url.getText().toString())));
    findViewById(R.id.open).setOnClickListener(v -> {
      Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
          .setType("*/*");
      startActivityForResult(i, OPEN);
    });
    handleIntent(getIntent());
    ui.post(this::refresh);
  }

  @Override
  protected void onNewIntent(Intent intent) {
    super.onNewIntent(intent);
    handleIntent(intent);
  }

  private void handleIntent(@Nullable Intent intent) {
    if (intent != null && Intent.ACTION_VIEW.equals(intent.getAction()) && intent.getData() != null) {
      play(intent.getData());
    }
  }

  @Override
  protected void onActivityResult(int req, int res, @Nullable Intent data) {
    super.onActivityResult(req, res, data);
    if (req == OPEN && res == RESULT_OK && data != null && data.getData() != null) play(data.getData());
  }

  private void play(Uri uri) {
    lastError = "";
    player.setMediaItem(MediaItem.fromUri(uri));
    player.prepare();
    player.play();
  }

  private void refresh() {
    String dec;
    try {
      dec = MimiLibrary.isAvailable() ? MimiLibrary.acquireIfReady() : "native library missing";
    } catch (MimiDecoderException e) {
      dec = "decoder error: " + e.getMessage();
    }
    status.setText(String.format(java.util.Locale.US, "position %.1f / %.1f s, state %d\n%s\n%s",
        player.getCurrentPosition() / 1e3, Math.max(0, player.getDuration()) / 1e3,
        player.getPlaybackState(), dec, lastError));
    ui.postDelayed(this::refresh, 500);
  }

  @Override
  protected void onDestroy() {
    ui.removeCallbacksAndMessages(null);
    player.release();
    super.onDestroy();
  }
}
