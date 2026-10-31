package org.joshjob.mimi;

import androidx.annotation.Nullable;
import androidx.media3.decoder.DecoderException;

/** Thrown when the Mimi decoder cannot be created or fails on a block. */
public final class MimiDecoderException extends DecoderException {
  public MimiDecoderException(String message) {
    super(message);
  }

  public MimiDecoderException(String message, @Nullable Throwable cause) {
    super(message, cause);
  }
}
