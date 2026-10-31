package org.joshjob.mimi;

import android.net.Uri;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.extractor.DefaultExtractorsFactory;
import androidx.media3.extractor.Extractor;
import androidx.media3.extractor.ExtractorsFactory;
import java.util.List;
import java.util.Map;

/** Wraps an {@link ExtractorsFactory} (default: {@link DefaultExtractorsFactory}), adding {@link MimiExtractor} first. */
@UnstableApi
public final class MimiExtractorsFactory implements ExtractorsFactory {
  private final ExtractorsFactory delegate;

  public MimiExtractorsFactory() {
    this(new DefaultExtractorsFactory());
  }

  public MimiExtractorsFactory(ExtractorsFactory delegate) {
    this.delegate = delegate;
  }

  @Override
  public Extractor[] createExtractors() {
    return prepend(delegate.createExtractors());
  }

  @Override
  public Extractor[] createExtractors(Uri uri, Map<String, List<String>> responseHeaders) {
    return prepend(delegate.createExtractors(uri, responseHeaders));
  }

  private static Extractor[] prepend(Extractor[] rest) {
    Extractor[] all = new Extractor[rest.length + 1];
    all[0] = new MimiExtractor();
    System.arraycopy(rest, 0, all, 1, rest.length);
    return all;
  }
}
