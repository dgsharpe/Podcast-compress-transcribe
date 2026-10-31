package org.joshjob.mimi;

import androidx.annotation.Nullable;
import androidx.media3.decoder.DecoderInputBuffer;
import androidx.media3.decoder.SimpleDecoder;
import androidx.media3.decoder.SimpleDecoderOutputBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Decodes the samples produced by {@link MimiExtractor} (one 1.44 s block each) to 16-bit mono PCM at
 * 24 kHz. Sample layout: int32 LE block index, valid frames, nq, frame bytes, then {@link
 * MimiExtractor#WINDOW} packed frames.
 */
public final class MimiDecoder
    extends SimpleDecoder<DecoderInputBuffer, SimpleDecoderOutputBuffer, MimiDecoderException> {
  public static final int SAMPLE_RATE = 24000;
  static final int SAMPLES_PER_FRAME = 1920;

  private final MimiLibrary.Handle handle;

  MimiDecoder(MimiLibrary.Handle handle, int numBuffers, int initialInputSize) {
    super(new DecoderInputBuffer[numBuffers], new SimpleDecoderOutputBuffer[numBuffers]);
    this.handle = handle;
    setInitialInputBufferSize(initialInputSize);
  }

  @Override
  public String getName() {
    return "libmimi(" + handle.backends + ")";
  }

  @Override
  protected DecoderInputBuffer createInputBuffer() {
    return new DecoderInputBuffer(DecoderInputBuffer.BUFFER_REPLACEMENT_MODE_DIRECT);
  }

  @Override
  protected SimpleDecoderOutputBuffer createOutputBuffer() {
    return new SimpleDecoderOutputBuffer(this::releaseOutputBuffer);
  }

  @Override
  protected MimiDecoderException createUnexpectedDecodeException(Throwable error) {
    return new MimiDecoderException("Unexpected decode error", error);
  }

  @Nullable
  @Override
  protected MimiDecoderException decode(
      DecoderInputBuffer inputBuffer, SimpleDecoderOutputBuffer outputBuffer, boolean reset) {
    if (reset) handle.reset();
    ByteBuffer in = inputBuffer.data;
    if (in == null || in.limit() < MimiExtractor.SAMPLE_HEADER) {
      return new MimiDecoderException("short sample");
    }
    ByteBuffer le = in.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    int block = le.getInt(0), valid = le.getInt(4), nq = le.getInt(8), frameBytes = le.getInt(12);
    if (in.limit() < MimiExtractor.SAMPLE_HEADER + MimiExtractor.WINDOW * frameBytes) {
      return new MimiDecoderException("truncated sample for block " + block);
    }
    ByteBuffer out = outputBuffer.init(inputBuffer.timeUs, valid * SAMPLES_PER_FRAME * 2);
    int n = handle.decode(this, block, in, MimiExtractor.SAMPLE_HEADER, nq, frameBytes, valid, out);
    if (n < 0) return new MimiDecoderException("native decode failed at block " + block);
    out.position(0);
    out.limit(n);
    return null;
  }
}
