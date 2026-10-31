package org.joshjob.mimi;

import android.content.Context;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Random-access decoding of one `.mimi` file to 16-bit PCM (24 kHz mono), independent of Media3.
 *
 * <p>Output is identical to straight-through decoding whatever the access order: block k is crossfaded
 * with block k−1's lookahead, so when k is requested without k−1 just before it, k−1 is decoded first
 * (and cached). A small LRU cache of decoded blocks serves repeated reads.
 */
public final class MimiFileDecoder implements AutoCloseable {
  public static final int SAMPLES_PER_BLOCK = MimiExtractor.BLOCK * MimiDecoder.SAMPLES_PER_FRAME;
  private static final int CACHE_BLOCKS = 16;  // 23 s of audio, 1.5 MB

  private final FileChannel ch;
  private final MimiLibrary.Handle handle;
  public final int nq, frameBytes, nFrames, nBlocks;
  public final long dataOffset;
  private final ByteBuffer window, pcm;
  private final Map<Integer, short[]> cache =
      new LinkedHashMap<Integer, short[]>(CACHE_BLOCKS, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<Integer, short[]> e) {
          return size() > CACHE_BLOCKS;
        }
      };

  public MimiFileDecoder(Context context, FileChannel channel) throws IOException, MimiDecoderException {
    this.ch = channel;
    ByteBuffer h = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    readFully(h, 0);
    if (h.get(0) != 'M' || h.get(1) != 'I' || h.get(2) != 'M' || h.get(3) != 'I' || h.get(4) != 1
        || h.get(6) != 11) {
      throw new IOException("not a v1 .mimi file");
    }
    nq = h.get(5) & 0xff;
    nFrames = h.getInt(16);
    frameBytes = h.getInt(20);
    dataOffset = h.getInt(24) & 0xffffffffL;
    if (nq < 1 || nq > 32 || frameBytes != (nq * 11 + 7) / 8) throw new IOException("bad header");
    nBlocks = (nFrames + MimiExtractor.BLOCK - 1) / MimiExtractor.BLOCK;
    handle = MimiLibrary.acquire(context);
    window = ByteBuffer.allocateDirect(MimiExtractor.WINDOW * frameBytes);
    pcm = ByteBuffer.allocateDirect(SAMPLES_PER_BLOCK * 2).order(ByteOrder.nativeOrder());
  }

  /** Total decoded length in samples. */
  public long numSamples() {
    return (long) nFrames * MimiDecoder.SAMPLES_PER_FRAME;
  }

  /** PCM of block k (SAMPLES_PER_BLOCK samples, fewer for the last block). */
  public synchronized short[] block(int k) throws IOException {
    short[] hit = cache.get(k);
    if (hit != null) return hit;
    if (k > 0 && !handle.follows(this, k)) decode(k - 1);  // prime block k's crossfade
    return decode(k);
  }

  private short[] decode(int k) throws IOException {
    int valid = Math.min(MimiExtractor.BLOCK, nFrames - k * MimiExtractor.BLOCK);
    // gather the clamped 25-frame window; consecutive frames are read in one go
    window.clear();
    int first = k * MimiExtractor.BLOCK - MimiExtractor.CONTEXT;
    for (int t = 0; t < MimiExtractor.WINDOW; ) {
      int f = Math.max(0, Math.min(nFrames - 1, first + t));
      int run = 1;
      while (t + run < MimiExtractor.WINDOW && first + t + run == f + run && f + run < nFrames) run++;
      ByteBuffer slice = window.duplicate();
      slice.position(t * frameBytes).limit((t + run) * frameBytes);
      readFully(slice, dataOffset + (long) f * frameBytes);
      t += run;
    }
    if (!handle.follows(this, k)) handle.reset();
    int n = handle.decode(this, k, window, 0, nq, frameBytes, valid, pcm);
    if (n < 0) throw new IOException("decode failed at block " + k);
    short[] out = new short[n / 2];
    pcm.position(0);
    pcm.asShortBuffer().get(out);
    cache.put(k, out);
    return out;
  }

  private void readFully(ByteBuffer b, long pos) throws IOException {
    while (b.hasRemaining()) {
      int r = ch.read(b, pos);
      if (r < 0) throw new IOException("unexpected end of file");
      pos += r;
    }
  }

  @Override
  public void close() throws IOException {
    ch.close();
  }
}
