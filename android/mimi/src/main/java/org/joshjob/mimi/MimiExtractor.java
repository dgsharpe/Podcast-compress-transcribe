package org.joshjob.mimi;

import androidx.annotation.Nullable;
import androidx.media3.common.C;
import androidx.media3.common.Format;
import androidx.media3.common.Metadata;
import androidx.media3.common.ParserException;
import androidx.media3.common.util.ParsableByteArray;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.extractor.Extractor;
import androidx.media3.extractor.ExtractorInput;
import androidx.media3.extractor.ExtractorOutput;
import androidx.media3.extractor.PositionHolder;
import androidx.media3.extractor.SeekMap;
import androidx.media3.extractor.SeekPoint;
import androidx.media3.extractor.TrackOutput;
import androidx.media3.extractor.metadata.id3.TextInformationFrame;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/**
 * Extracts `.mimi` files (FORMAT.md). Each output sample is one decode block: 18 new frames (1.44 s)
 * plus 6 frames of left context and 1 of lookahead, so every sample decodes on its own and seeks are
 * exact. Frames are read sequentially; the 7 frames shared with the next block are kept in memory.
 */
@UnstableApi
public final class MimiExtractor implements Extractor {
  static final int CONTEXT = 6, BLOCK = 18, LOOKAHEAD = 1, WINDOW = CONTEXT + BLOCK + LOOKAHEAD;
  static final int SAMPLE_HEADER = 16;
  static final long FRAME_US = 80_000;
  static final long BLOCK_US = BLOCK * FRAME_US;
  private static final int HEADER = 64;

  private @Nullable ExtractorOutput output;
  private @Nullable TrackOutput track;
  private boolean headerDone;
  private int nq, frameBytes, nFrames, nBlocks;
  private long dataOffset;
  // frames [cacheStart, cacheStart + cacheCount) are held in `cache`
  private byte[] cache = new byte[0];
  private int cacheStart, cacheCount;
  private int nextBlock;
  private boolean atFileStart = true;  // input is positioned at byte 0 (header still to be consumed)

  @Override
  public boolean sniff(ExtractorInput input) throws IOException {
    byte[] h = new byte[8];
    if (!input.peekFully(h, 0, 8, /* allowEndOfInput= */ true)) return false;
    return h[0] == 'M' && h[1] == 'I' && h[2] == 'M' && h[3] == 'I' && h[4] == 1;
  }

  @Override
  public void init(ExtractorOutput output) {
    this.output = output;
  }

  @Override
  public int read(ExtractorInput input, PositionHolder seekPosition) throws IOException {
    if (atFileStart) {
      if (headerDone) input.skipFully((int) dataOffset);  // restart from 0: header already parsed
      else readHeader(input);
      atFileStart = false;
      return RESULT_CONTINUE;
    }
    if (nextBlock >= nBlocks) return RESULT_END_OF_INPUT;
    int k = nextBlock;
    int first = Math.max(0, k * BLOCK - CONTEXT);
    int last = Math.min(nFrames, k * BLOCK + BLOCK + LOOKAHEAD);  // exclusive
    if (cacheCount == 0) cacheStart = first;
    while (cacheStart + cacheCount < last) {  // read the missing frames
      ensureCapacity(cacheCount + 1);
      if (!input.readFully(cache, cacheCount * frameBytes, frameBytes, /* allowEndOfInput= */ true)) {
        return RESULT_END_OF_INPUT;
      }
      cacheCount++;
    }
    emit(k);
    nextBlock++;
    int drop = Math.max(0, Math.min(cacheCount, (k + 1) * BLOCK - CONTEXT - cacheStart));
    if (drop > 0) {
      System.arraycopy(cache, drop * frameBytes, cache, 0, (cacheCount - drop) * frameBytes);
      cacheStart += drop;
      cacheCount -= drop;
    }
    return RESULT_CONTINUE;
  }

  private void emit(int k) {
    int valid = Math.min(BLOCK, nFrames - k * BLOCK);
    byte[] s = new byte[SAMPLE_HEADER + WINDOW * frameBytes];
    putInt(s, 0, k);
    putInt(s, 4, valid);
    putInt(s, 8, nq);
    putInt(s, 12, frameBytes);
    for (int t = 0; t < WINDOW; t++) {  // clamp to [0, nFrames)
      int f = Math.max(0, Math.min(nFrames - 1, k * BLOCK - CONTEXT + t));
      System.arraycopy(cache, (f - cacheStart) * frameBytes, s, SAMPLE_HEADER + t * frameBytes,
          frameBytes);
    }
    TrackOutput out = track;
    out.sampleData(new ParsableByteArray(s), s.length);
    out.sampleMetadata(k * BLOCK_US, C.BUFFER_FLAG_KEY_FRAME, s.length, 0, null);
  }

  private void readHeader(ExtractorInput input) throws IOException {
    byte[] h = new byte[HEADER];
    input.readFully(h, 0, HEADER);
    if (!(h[0] == 'M' && h[1] == 'I' && h[2] == 'M' && h[3] == 'I') || h[4] != 1 || h[6] != 11) {
      throw ParserException.createForMalformedContainer("not a v1 .mimi file", null);
    }
    nq = h[5] & 0xff;
    nFrames = getInt(h, 16);
    frameBytes = getInt(h, 20);
    dataOffset = getInt(h, 24) & 0xffffffffL;
    int metaLen = getInt(h, 28);
    if (nq < 1 || nq > 32 || frameBytes != (nq * 11 + 7) / 8 || dataOffset < HEADER + metaLen) {
      throw ParserException.createForMalformedContainer("bad .mimi header", null);
    }
    byte[] meta = new byte[metaLen];
    input.readFully(meta, 0, metaLen);
    input.skipFully((int) (dataOffset - HEADER - metaLen));
    nBlocks = (nFrames + BLOCK - 1) / BLOCK;

    Format.Builder fb =
        new Format.Builder()
            .setSampleMimeType(MimiLibrary.MIME_TYPE)
            .setContainerMimeType(MimiLibrary.MIME_TYPE)
            .setSampleRate(MimiDecoder.SAMPLE_RATE)
            .setChannelCount(1)
            .setAverageBitrate((int) (frameBytes * 8 * 12.5))
            .setMaxInputSize(SAMPLE_HEADER + WINDOW * frameBytes)
            .setInitializationData(ImmutableList.of(h));
    String title = titleFrom(meta);
    if (title != null) {
      fb.setMetadata(
          new Metadata(new TextInformationFrame("TIT2", null, ImmutableList.of(title))));
    }
    ExtractorOutput out = output;
    track = out.track(0, C.TRACK_TYPE_AUDIO);
    track.format(fb.build());
    out.endTracks();
    out.seekMap(new BlockSeekMap());
    headerDone = true;
  }

  @Nullable
  private static String titleFrom(byte[] meta) {
    try {
      String t = new JSONObject(new String(meta, StandardCharsets.UTF_8)).optString("title", "");
      return t.isEmpty() ? null : t;
    } catch (Exception e) {
      return null;
    }
  }

  /**
   * Media3 reopens the input at {@code position} (a {@link BlockSeekMap} seek point, or 0) and passes
   * the seek point's time, which is the start of the block whose window begins at {@code position}.
   */
  @Override
  public void seek(long position, long timeUs) {
    cacheCount = 0;
    atFileStart = position == 0;
    nextBlock = atFileStart ? 0 : blockAt(timeUs);
  }

  private int blockAt(long timeUs) {
    return (int) Math.max(0, Math.min(nBlocks - 1, timeUs / BLOCK_US));
  }

  private long blockPosition(int k) {
    return dataOffset + (long) Math.max(0, k * BLOCK - CONTEXT) * frameBytes;
  }

  @Override
  public void release() {}

  private void ensureCapacity(int frames) {
    if (cache.length < frames * frameBytes) {
      byte[] c = new byte[Math.max(frames, 2 * WINDOW) * frameBytes];
      System.arraycopy(cache, 0, c, 0, cacheCount * frameBytes);
      cache = c;
    }
  }

  private static int getInt(byte[] b, int o) {
    return (b[o] & 0xff) | (b[o + 1] & 0xff) << 8 | (b[o + 2] & 0xff) << 16 | (b[o + 3] & 0xff) << 24;
  }

  private static void putInt(byte[] b, int o, int v) {
    b[o] = (byte) v;
    b[o + 1] = (byte) (v >> 8);
    b[o + 2] = (byte) (v >> 16);
    b[o + 3] = (byte) (v >> 24);
  }

  private final class BlockSeekMap implements SeekMap {
    @Override
    public boolean isSeekable() {
      return true;
    }

    @Override
    public long getDurationUs() {
      return nFrames * FRAME_US;
    }

    @Override
    public SeekPoints getSeekPoints(long timeUs) {
      int k = blockAt(timeUs);
      return new SeekPoints(new SeekPoint(k * BLOCK_US, blockPosition(k)));
    }
  }
}
