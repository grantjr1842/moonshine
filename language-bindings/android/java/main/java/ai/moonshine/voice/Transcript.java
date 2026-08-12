package ai.moonshine.voice;

import java.util.List;

public class Transcript {
    public List<TranscriptLine> lines;
    /**
     * Monotonically increasing snapshot revision from the native
     * transcript_t struct. A-147: callers should pass this to
     * {@link JNI#moonshineStreamAcknowledgeRevision} after consuming
     * the transcript so the server's clear_update_flags(up_to_revision)
     * stops re-flagging lines the client has already observed.
     */
    public long revision;
    public String text() {
        StringBuilder text = new StringBuilder();
        for (TranscriptLine line : lines) {
            text.append(line.text).append("\n");
        }
        return text.toString();
    }
}
