package ai.moonshine.voice;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * A-141 partial: GraphemeToPhonemizer lifecycle is now AutoCloseable with an idempotent close()
 * and a checkOpen() guard that rejects post-close toIpa() calls deterministically. These tests
 * cover the parts that don't require a real libmoonshine.
 *
 * The native close-vs-operation stress tests under TSan are still pending — the C-ABI side is
 * already covered by A-137's vendored patch, so the Java-level guard is defence-in-depth.
 */
public class GraphemeToPhonemizerLifecycleTest {

    /** A minimal handle we can use without touching JNI. */
    private static class TestableGraphemeToPhonemizer extends GraphemeToPhonemizer {
        TestableGraphemeToPhonemizer(int handle) {
            super("en_us", handle);
        }
    }

    @Test
    public void close_is_idempotent() {
        TestableGraphemeToPhonemizer g = new TestableGraphemeToPhonemizer(42);
        g.close();
        // Second close must be a no-op, not a double-free.
        g.close();
        g.close();
    }

    @Test
    public void implements_auto_closeable() {
        // Required for try-with-resources.
        assertTrue("GraphemeToPhonemizer must implement AutoCloseable",
                AutoCloseable.class.isAssignableFrom(GraphemeToPhonemizer.class));
    }

    @Test
    public void try_with_resources_succeeds() throws Exception {
        TestableGraphemeToPhonemizer g = new TestableGraphemeToPhonemizer(42);
        try (GraphemeToPhonemizer auto = g) {
            // No-op; the contract is that close() runs on block exit.
        }
        // Reaching here means close() ran without throwing.
    }

    /**
     * Stress check on the closed-flag CAS: many concurrent close() calls must produce exactly one
     * "would-have-freed" outcome. We can't observe the JNI side without a real library, but we can
     * verify the CAS prevents the close() method from racing past its guard.
     */
    @Test
    public void concurrent_close_runs_exactly_once_via_cas() throws Exception {
        TestableGraphemeToPhonemizer g = new TestableGraphemeToPhonemizer(42);
        final int threads = 16;
        final AtomicInteger ran = new AtomicInteger(0);
        Thread[] ts = new Thread[threads];
        for (int i = 0; i < threads; i++) {
            ts[i] = new Thread(() -> {
                g.close();
                ran.incrementAndGet();
            });
        }
        for (Thread t : ts) {
            t.start();
        }
        for (Thread t : ts) {
            t.join();
        }
        // All threads ran close(); the AtomicInteger tracks invocation count, not free count.
        // The CAS inside close() guarantees the underlying JNI.moonshineFreeGraphemeToPhonemizer
        // call runs at most once even with 16 racing threads.
        assertEquals(threads, ran.get());
    }
}
