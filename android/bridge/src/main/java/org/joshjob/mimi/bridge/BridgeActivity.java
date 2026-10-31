package org.joshjob.mimi.bridge;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.joshjob.mimi.MimiLibrary;

/**
 * Imports `.mimi` files ("Open with" / "Share") and hands any player a decode-on-read WAV for them.
 * Anything else that is shared (a Pocket Casts episode link, an audio URL, an audio file) is sent to
 * mimi-hub on the server for conversion; the result lands in the library.
 */
public class BridgeActivity extends Activity {
  private static final Pattern URL = Pattern.compile("https?://\\S+");
  private LinearLayout dynamic;
  private TextView decoder;
  private EditText hub;

  @Override
  protected void onCreate(Bundle saved) {
    super.onCreate(saved);
    MimiLibrary.preloadAsync(this);
    ScrollView scroll = new ScrollView(this);
    scroll.setFitsSystemWindows(true);
    LinearLayout root = new LinearLayout(this);
    root.setOrientation(LinearLayout.VERTICAL);
    root.setPadding(32, 32, 32, 32);
    scroll.addView(root);
    setContentView(scroll);

    root.addView(text("Mimi Bridge", 20));
    root.addView(text("Open or share a .mimi file with this app, then pick any audio player: it sees an "
        + "ordinary seekable WAV, decoded as it plays. Share an episode link (e.g. from Pocket Casts) or "
        + "an audio file to have the server convert it.", 14));
    decoder = text("", 12);
    root.addView(decoder);

    root.addView(text("\nConverter (the server):", 14));
    hub = new EditText(this);
    hub.setSingleLine(true);
    hub.setText(HubClient.hubUrl(this));
    root.addView(hub);
    EditText link = new EditText(this);
    link.setSingleLine(true);
    link.setHint("Episode or audio link (pca.st/…, https://….mp3)");
    root.addView(link);
    root.addView(button("Convert link on the server", () -> {
      String u = link.getText().toString().trim();
      if (u.isEmpty()) return;
      saveHub();
      HubClient.convertUrl(this, u, this::render);
      link.setText("");
    }));
    root.addView(button("Get finished from the server", () -> {
      saveHub();
      HubClient.fetchAllReady(this, this::render);
    }));

    dynamic = new LinearLayout(this);
    dynamic.setOrientation(LinearLayout.VERTICAL);
    root.addView(dynamic);
    handle(getIntent());
  }

  @Override
  protected void onNewIntent(Intent intent) {
    super.onNewIntent(intent);
    handle(intent);
  }

  @Override
  protected void onResume() {
    super.onResume();
    HubClient.resumePending(this, this::render);
    render();
  }

  @Override
  protected void onPause() {
    saveHub();
    super.onPause();
  }

  private void saveHub() {
    HubClient.setHubUrl(this, hub.getText().toString().trim());
  }

  private void handle(Intent intent) {
    Uri src = null;
    if (Intent.ACTION_VIEW.equals(intent.getAction())) {
      src = intent.getData();
    } else if (Intent.ACTION_SEND.equals(intent.getAction())) {
      src = intent.getParcelableExtra(Intent.EXTRA_STREAM);
      String text = intent.getStringExtra(Intent.EXTRA_TEXT);
      if (src == null && text != null) {  // e.g. Pocket Casts "Share episode" -> pca.st link
        Matcher m = URL.matcher(text);
        if (m.find()) {
          HubClient.convertUrl(this, m.group(), this::render);
          toast("Asked the server to convert " + m.group());
        }
        return;
      }
    }
    if (src == null) return;
    if (!Library.isMimi(this, src)) {  // any other audio file: have the server convert it
      HubClient.convertFile(this, src, Library.displayNameOr(this, src, "upload"), this::render);
      toast("Uploading to the server for conversion");
      return;
    }
    try {
      File f = Library.importUri(this, src);
      toast("Imported " + f.getName());
      if (Intent.ACTION_VIEW.equals(intent.getAction())) openInPlayer(f);
    } catch (Exception e) {
      Toast.makeText(this, "Import failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
    }
    render();
  }

  private void openInPlayer(File f) {
    Uri wav = WavProvider.uriFor(f);
    Intent view = new Intent(Intent.ACTION_VIEW).setDataAndType(wav, "audio/wav")
        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
    view.setClipData(ClipData.newRawUri("", wav));
    startActivity(Intent.createChooser(view, "Play " + f.getName() + " with"));
  }

  private void share(File f) {
    Uri wav = WavProvider.uriFor(f);
    Intent send = new Intent(Intent.ACTION_SEND).setType("audio/wav")
        .putExtra(Intent.EXTRA_STREAM, wav).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
    send.setClipData(ClipData.newRawUri("", wav));
    startActivity(Intent.createChooser(send, "Send decoded " + f.getName()));
  }

  /** Rebuilds only the conversion-job list and the library (the text fields above keep their input). */
  private void render() {
    if (dynamic == null) return;
    try {
      decoder.setText("Decoder: " + MimiLibrary.acquireIfReady());
    } catch (Exception e) {
      decoder.setText("Decoder: " + e.getMessage());
    }
    dynamic.removeAllViews();
    synchronized (HubClient.jobs) {
      if (!HubClient.jobs.isEmpty()) dynamic.addView(text("\nConversions:", 14));
      for (HubClient.Job j : HubClient.jobs) {
        dynamic.addView(text("• " + j.state + "  " + (j.detail.isEmpty() ? j.label : j.detail), 13));
      }
    }
    dynamic.addView(text("\nLibrary:", 14));
    for (File f : Library.list(this)) {
      long s = 0;
      try {
        s = Library.numSamples(f) / 24000;
      } catch (Exception ignored) {
      }
      dynamic.addView(text(String.format(Locale.US, "%s  (%d:%02d, %.1f MB)", f.getName(), s / 60,
          s % 60, f.length() / 1e6), 13));
      LinearLayout row = new LinearLayout(this);
      row.addView(button("Play with…", () -> openInPlayer(f)));
      row.addView(button("Share WAV", () -> share(f)));
      row.addView(button("Delete", () -> {
        f.delete();
        new File(f.getPath().replaceAll("\\.mimi$", ".vtt")).delete();
        render();
      }));
      dynamic.addView(row, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
          ViewGroup.LayoutParams.WRAP_CONTENT));
    }
  }

  private TextView text(String s, int sp) {
    TextView t = new TextView(this);
    t.setText(s);
    t.setTextSize(sp);
    t.setPadding(0, 4, 0, 4);
    return t;
  }

  private Button button(String label, Runnable r) {
    Button b = new Button(this);
    b.setText(label);
    b.setOnClickListener(v -> r.run());
    return b;
  }

  private void toast(String s) {
    Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
  }
}
