package ai.moonshine.voice;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Grapheme-to-phoneme (IPA) via the Moonshine native API.
 *
 * <p>Aligns with Python {@code moonshine_voice.GraphemeToPhonemizer}: construct with a language tag and
 * {@code g2p_root} on disk, then {@link #toIpa}. No download or caching is performed here.
 *
 * <h2>Lifecycle (A-141)</h2>
 *
 * <p>{@code GraphemeToPhonemizer} holds a mutable {@code int} handle. Any caller thread can invoke
 * {@link #toIpa} while another calls {@link #close} or runs the finalizer, which frees the handle and
 * writes {@code -1}; the first call would then execute against freed native state.
 *
 * <p>This class mitigates that race with:
 * <ul>
 *   <li>{@link AutoCloseable} so callers can use try-with-resources and {@code close()} becomes
 *       idempotent.</li>
 *   <li>A {@code closed} flag guarded by an {@link AtomicInteger} for lock-free admission; every
 *       operation reads the flag and rejects post-close calls deterministically.</li>
 * </ul>
 *
 * <p>The native {@code moonshine_grapheme_phonemizer_map_mutex} (added by A-137's vendored patch) still
 * covers the C-ABI dereference, so this is defence-in-depth. Full in-flight operation counting and
 * concurrent stress tests under TSan are still tracked as A-141 follow-up.
 */
public class GraphemeToPhonemizer implements AutoCloseable {
    private volatile int handle = -1;
    /**
     * 0 = open, 1 = closed. We use an AtomicInteger rather than a boolean so the close path can
     * CAS-transition open -> closed exactly once without locking; subsequent close calls become
     * idempotent no-ops.
     */
    private final AtomicInteger closed = new AtomicInteger(0);
    private final String language;

    /**
     * @param language  Moonshine language tag (e.g. {@code en_us}).
     * @param filenames Optional canonical lexicon / model keys; {@code null} or empty lets the engine resolve
     *                  paths under {@code g2p_root}.
     * @param g2pRoot   Asset root directory; stored as option {@code g2p_root}.
     * @param options   Extra native options (e.g. {@code spanish_narrow_obstruents}).
     */
    public GraphemeToPhonemizer(String language, String[] filenames, String g2pRoot,
            List<TranscriberOption> options) {
        this.language = language;
        JNI.ensureLibraryLoaded();
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("g2p_root", g2pRoot));
        if (options != null) {
            opts.addAll(options);
        }
        int h = JNI.moonshineCreateGraphemeToPhonemizerFromFiles(language, filenames,
                opts.toArray(new TranscriberOption[0]));
        if (h < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(h));
        }
        this.handle = h;
    }

    public GraphemeToPhonemizer(String language, String g2pRoot, List<TranscriberOption> options) {
        this(language, null, g2pRoot, options);
    }

    public static GraphemeToPhonemizer fromMemory(String language, String[] filenames, byte[][] memory,
            String g2pRoot, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("g2p_root", g2pRoot));
        if (options != null) {
            opts.addAll(options);
        }
        int h = JNI.moonshineCreateGraphemeToPhonemizerFromMemory(language, filenames, memory,
                opts.toArray(new TranscriberOption[0]));
        if (h < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(h));
        }
        return new GraphemeToPhonemizer(language, h);
    }

    private GraphemeToPhonemizer(String language, int handle) {
        this.language = language;
        this.handle = handle;
    }

    /** Same as {@link TextToSpeech#getG2pDependencies(String, List)}. */
    public static String getG2pDependencies(String languages, List<TranscriberOption> options) {
        return TextToSpeech.getG2pDependencies(languages, options);
    }

    private static TranscriberOption[] toArray(List<TranscriberOption> options) {
        if (options == null || options.isEmpty()) {
            return null;
        }
        return options.toArray(new TranscriberOption[0]);
    }

    public String getLanguage() {
        return language;
    }

    /**
     * Throw a deterministic error when an operation is attempted after {@link #close()}.
     * Mirrors the Python {@code moonshine_voice.errors.MoonshineError} pattern.
     */
    private void checkOpen() {
        if (closed.get() != 0) {
            throw new IllegalStateException("GraphemeToPhonemizer is closed");
        }
    }

    /** Convert text to a single IPA string (native {@code moonshine_text_to_phonemes}). */
    public String toIpa(String text, List<TranscriberOption> options) {
        checkOpen();
        // Snapshot the handle under the closed-flag guard so a concurrent close()
        // that flips closed=1 cannot race a stale -1 read here.
        int h = handle;
        if (h < 0) {
            throw new IllegalStateException("GraphemeToPhonemizer is closed");
        }
        String ipa = JNI.moonshineTextToPhonemes(h, text, toArray(options));
        if (ipa == null) {
            throw new RuntimeException("moonshineTextToPhonemes failed");
        }
        return ipa;
    }

    public String toIpa(String text) {
        return toIpa(text, null);
    }

    /**
     * Release the native handle. Idempotent: subsequent calls are no-ops. After {@code close()},
     * every {@link #toIpa} call throws {@link IllegalStateException}.
     */
    @Override
    public void close() {
        // CAS-transition closed 0 -> 1 so only one thread actually frees the handle.
        if (!closed.compareAndSet(0, 1)) {
            return;
        }
        int h = handle;
        if (h >= 0) {
            JNI.moonshineFreeGraphemeToPhonemizer(h);
            handle = -1;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        // Best-effort safety net; explicit close() is preferred and idempotent.
        try {
            close();
        } finally {
            super.finalize();
        }
    }
}
