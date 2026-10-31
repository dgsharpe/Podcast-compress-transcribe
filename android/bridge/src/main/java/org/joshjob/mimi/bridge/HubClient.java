package org.joshjob.mimi.bridge;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/**
 * Talks to mimi-hub on the server (hub/mimi_hub.py): requests conversions, polls them, and pulls finished
 * `.mimi` + `.vtt` files into the {@link Library}. Jobs are process-wide so they survive the activity.
 */
final class HubClient {
  static final String DEFAULT_URL = "http://<tailscale-ip>:8765";  // the server on Tailscale

  /** A conversion this app asked for. */
  static final class Job {
    final String label;
    volatile String id = "", state = "requesting", detail = "";

    Job(String label) {
      this.label = label;
    }
  }

  static final List<Job> jobs = new ArrayList<>();
  private static final ExecutorService pool = Executors.newCachedThreadPool();
  private static final Handler main = new Handler(Looper.getMainLooper());

  static String hubUrl(Context c) {
    return prefs(c).getString("hub", DEFAULT_URL);
  }

  static void setHubUrl(Context c, String url) {
    prefs(c).edit().putString("hub", url.replaceAll("/+$", "")).apply();
  }

  private static SharedPreferences prefs(Context c) {
    return c.getSharedPreferences("bridge", Context.MODE_PRIVATE);
  }

  /** Ask the hub to convert a URL (direct audio, or a Pocket Casts share link). */
  static void convertUrl(Context c, String url, Runnable changed) {
    Job j = add(url, changed);
    String hub = hubUrl(c);
    Context app = c.getApplicationContext();
    pool.execute(() -> {
      try {
        JSONObject body = new JSONObject().put("url", url);
        JSONObject r = request("POST", hub + "/v1/convert", "application/json",
            body.toString().getBytes(StandardCharsets.UTF_8));
        follow(app, hub, j, r, changed);
      } catch (Exception e) {
        fail(j, e, changed);
      }
    });
  }

  /** Upload a local audio file to the hub for conversion. */
  static void convertFile(Context c, Uri uri, String name, Runnable changed) {
    Job j = add(name, changed);
    String hub = hubUrl(c);
    Context app = c.getApplicationContext();
    pool.execute(() -> {
      try {
        String boundary = "mimi" + System.nanoTime();
        HttpURLConnection h = open(hub + "/v1/upload?title=" + Uri.encode(name), "POST");
        h.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        h.setChunkedStreamingMode(1 << 16);
        try (OutputStream out = h.getOutputStream();
            InputStream in = app.getContentResolver().openInputStream(uri)) {
          out.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\""
              + name.replace("\"", "") + "\"\r\nContent-Type: application/octet-stream\r\n\r\n")
              .getBytes(StandardCharsets.UTF_8));
          byte[] buf = new byte[1 << 16];
          int n;
          while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
          out.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        }
        follow(app, hub, j, readJson(h), changed);
      } catch (Exception e) {
        fail(j, e, changed);
      }
    });
  }

  /**
   * Polls job j until it is ready, then downloads it. Network errors (e.g. the phone sleeping) are
   * retried for up to 10 minutes; the id stays in the pending set either way, so {@link
   * #resumePending} picks it up the next time the app is opened.
   */
  private static void follow(Context app, String hub, Job j, JSONObject r, Runnable changed)
      throws Exception {
    j.id = r.getString("id");
    setPending(app, j.id, true);
    long lastOk = System.currentTimeMillis();
    while (true) {
      if (r != null) {
        lastOk = System.currentTimeMillis();
        j.state = r.getString("state");
        String title = r.optString("title", "");
        j.detail = (title.isEmpty() ? "" : title + " — ")
            + (r.isNull("error") ? "" : r.optString("error"))
            + ("encoding".equals(j.state) ? String.format(java.util.Locale.US, " %.0f%%", 100 * r.optDouble("progress", 0)) : "");
        post(changed);
        if ("ready".equals(j.state)) break;
        if ("failed".equals(j.state)) {
          setPending(app, j.id, false);
          return;
        }
      } else if (System.currentTimeMillis() - lastOk > 10 * 60_000) {
        j.state = "paused";
        j.detail = "lost contact with the server; resumes when you reopen Mimi Bridge";
        post(changed);
        return;
      }
      Thread.sleep(3000);
      try {
        r = request("GET", hub + "/v1/episodes/" + j.id, null, null);
      } catch (IOException offline) {
        r = null;
        j.detail = "waiting for network…";
        post(changed);
      }
    }
    download(app, hub, j, r, changed);
  }

  private static void download(Context app, String hub, Job j, JSONObject r, Runnable changed)
      throws Exception {
    j.state = "downloading result";
    post(changed);
    String base = safeName(r.optString("title", ""), j.id);
    fetch(hub + "/v1/episodes/" + j.id + ".mimi", new File(Library.dir(app), base + ".mimi"));
    try {
      fetch(hub + "/v1/episodes/" + j.id + ".vtt", new File(Library.dir(app), base + ".vtt"));
    } catch (IOException noTranscript) {
    }
    setPending(app, j.id, false);
    j.state = "done";
    j.detail = base + ".mimi";
    post(changed);
  }

  /** Re-follows jobs requested earlier that never finished downloading (app killed, phone slept). */
  static void resumePending(Context c, Runnable changed) {
    Context app = c.getApplicationContext();
    String hub = hubUrl(c);
    for (String id : prefs(c).getStringSet("pending", new java.util.HashSet<>())) {
      synchronized (jobs) {
        boolean active = false;
        for (Job j : jobs) {
          active |= id.equals(j.id) && !"failed".equals(j.state) && !"paused".equals(j.state);
        }
        if (active) continue;
      }
      Job j = add(id, changed);
      j.id = id;
      pool.execute(() -> {
        try {
          follow(app, hub, j, request("GET", hub + "/v1/episodes/" + id, null, null), changed);
        } catch (Exception e) {
          fail(j, e, changed);
        }
      });
    }
  }

  /** Downloads every ready episode on the server that isn't in the library yet. */
  static void fetchAllReady(Context c, Runnable changed) {
    Context app = c.getApplicationContext();
    String hub = hubUrl(c);
    pool.execute(() -> {
      try {
        org.json.JSONArray all = new org.json.JSONArray(new String(readAll(
            open(hub + "/v1/episodes", "GET").getInputStream()), StandardCharsets.UTF_8));
        for (int i = 0; i < all.length(); i++) {
          JSONObject r = all.getJSONObject(i);
          String id = r.getString("id");
          if (!"ready".equals(r.optString("state"))) continue;
          if (new File(Library.dir(app), safeName(r.optString("title", ""), id) + ".mimi").exists()) continue;
          Job j = add(r.optString("title", id), changed);
          j.id = id;
          try {
            download(app, hub, j, r, changed);
          } catch (Exception e) {
            fail(j, e, changed);
          }
        }
      } catch (Exception e) {
        fail(add("fetch from the server", changed), e, changed);
      }
    });
  }

  private static synchronized void setPending(Context c, String id, boolean on) {
    java.util.Set<String> s = new java.util.HashSet<>(prefs(c).getStringSet("pending", new java.util.HashSet<>()));
    if (on) s.add(id);
    else s.remove(id);
    prefs(c).edit().putStringSet("pending", s).apply();
  }

  private static Job add(String label, Runnable changed) {
    Job j = new Job(label);
    synchronized (jobs) {
      jobs.add(0, j);
    }
    post(changed);
    return j;
  }

  private static void fail(Job j, Exception e, Runnable changed) {
    j.state = "failed";
    j.detail = e.getClass().getSimpleName() + ": " + e.getMessage();
    post(changed);
  }

  private static void post(Runnable r) {
    if (r != null) main.post(r);
  }

  static String safeName(String title, String id) {
    String s = title.replaceAll("[^A-Za-z0-9._ -]+", "").trim().replaceAll("\\s+", "_");
    if (s.length() > 80) s = s.substring(0, 80);
    return s.isEmpty() ? id : s + "_" + id.substring(0, Math.min(8, id.length()));
  }

  private static HttpURLConnection open(String url, String method) throws IOException {
    HttpURLConnection h = (HttpURLConnection) new URL(url).openConnection();
    h.setRequestMethod(method);
    h.setConnectTimeout(15000);
    h.setReadTimeout(60000);
    if ("POST".equals(method)) h.setDoOutput(true);
    return h;
  }

  private static JSONObject request(String method, String url, String type, byte[] body)
      throws Exception {
    HttpURLConnection h = open(url, method);
    if (body != null) {
      h.setRequestProperty("Content-Type", type);
      try (OutputStream o = h.getOutputStream()) {
        o.write(body);
      }
    }
    return readJson(h);
  }

  private static JSONObject readJson(HttpURLConnection h) throws Exception {
    int code = h.getResponseCode();
    InputStream in = code < 400 ? h.getInputStream() : h.getErrorStream();
    String s = in == null ? "" : new String(readAll(in), StandardCharsets.UTF_8);
    if (code >= 400) throw new IOException("HTTP " + code + " " + s);
    return new JSONObject(s);
  }

  private static void fetch(String url, File dst) throws IOException {
    HttpURLConnection h = open(url, "GET");
    if (h.getResponseCode() != 200) throw new IOException("HTTP " + h.getResponseCode() + " for " + url);
    File tmp = new File(dst.getPath() + ".tmp");
    try (InputStream in = h.getInputStream(); OutputStream out = new FileOutputStream(tmp)) {
      byte[] buf = new byte[1 << 16];
      int n;
      while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
    }
    if (!tmp.renameTo(dst)) throw new IOException("rename failed");
  }

  private static byte[] readAll(InputStream in) throws IOException {
    java.io.ByteArrayOutputStream b = new java.io.ByteArrayOutputStream();
    byte[] buf = new byte[8192];
    int n;
    while ((n = in.read(buf)) > 0) b.write(buf, 0, n);
    in.close();
    return b.toByteArray();
  }
}
