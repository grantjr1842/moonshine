// Per-stream delta semantics regression for A-147. Exercises
// TranscriptStreamOutput directly so the timer-update-then-poll bug
// is covered without loading a streaming model.

#include "transcriber.h"

#include <algorithm>
#include <string>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

TranscriberLine make_line(uint64_t id, const std::string &text) {
  TranscriberLine line;
  line.id = id;
  line.is_complete = true;
  line.start_time = 0.0f;
  line.duration = 1.0f;
  line.text = new std::string(text);
  line.has_text_changed = false;
  line.just_updated = true;
  return line;
}

void free_line_text(TranscriberLine &line) {
  delete line.text;
  line.text = nullptr;
}

void push_line(TranscriptStreamOutput &output, TranscriberLine &line) {
  output.add_or_update_line(line);
  if (std::find(output.ordered_internal_line_ids.begin(),
                 output.ordered_internal_line_ids.end(),
                 line.id) == output.ordered_internal_line_ids.end()) {
    output.ordered_internal_line_ids.push_back(line.id);
  }
}

}  // namespace

TEST_CASE("streaming delta: per-line revision advances on add") {
  TranscriptStreamOutput output;

  TranscriberLine hello = make_line(1, "Hello");
  push_line(output, hello);
  CHECK(output.internal_lines_map.at(1).revision == 1);
  TranscriberLine hello_world = make_line(1, "Hello world");
  push_line(output, hello_world);
  CHECK(output.internal_lines_map.at(1).revision == 2);
  CHECK(output.revision == 2);

  free_line_text(output.internal_lines_map.at(1));
}

TEST_CASE("streaming delta: timer-update-then-poll surfaces text change") {
  TranscriptStreamOutput output;

  TranscriberLine first = make_line(1, "first");
  push_line(output, first);
  output.update_transcript_from_lines();
  const uint64_t revision_after_first_update = output.revision;

  TranscriberLine second = make_line(1, "second");
  push_line(output, second);
  // After a timer update + poll sequence, the new revision must be
  // observed_by_client == false on the un-acked snapshot.
  CHECK(output.internal_lines_map.at(1).revision >
        revision_after_first_update);
  output.update_transcript_from_lines();
  CHECK(output.output_lines.at(0).has_text_changed == 1);
  CHECK(std::string(output.output_lines.at(0).text) == "second");

  free_line_text(output.internal_lines_map.at(1));
}

TEST_CASE("streaming delta: clear_update_flags is bounded by revision") {
  TranscriptStreamOutput output;

  TranscriberLine first = make_line(1, "first");
  push_line(output, first);
  output.update_transcript_from_lines();
  const uint64_t first_revision = output.revision;

  TranscriberLine second = make_line(1, "second");
  push_line(output, second);
  output.update_transcript_from_lines();
  const uint64_t second_revision = output.revision;
  CHECK(second_revision > first_revision);

  output.clear_update_flags(first_revision);
  output.update_transcript_from_lines();
  CHECK(output.output_lines.at(0).has_text_changed == 1);
  CHECK(std::string(output.output_lines.at(0).text) == "second");

  output.last_acknowledged_revision = second_revision;
  output.clear_update_flags(second_revision);
  output.update_transcript_from_lines();
  CHECK(output.output_lines.at(0).has_text_changed == 0);

  free_line_text(output.internal_lines_map.at(1));
}

TEST_CASE("streaming delta: stale acknowledge does not regress") {
  TranscriptStreamOutput output;

  TranscriberLine first = make_line(1, "first");
  push_line(output, first);
  output.update_transcript_from_lines();
  const uint64_t first_revision = output.revision;

  output.last_acknowledged_revision = first_revision;

  TranscriberLine second = make_line(1, "second");
  push_line(output, second);
  output.update_transcript_from_lines();
  const uint64_t second_revision = output.revision;
  CHECK(second_revision > first_revision);

  // Stale ack with a lower revision must not regress the high-water mark.
  output.last_acknowledged_revision = first_revision;
  output.clear_update_flags(output.last_acknowledged_revision);
  output.update_transcript_from_lines();
  CHECK(output.output_lines.at(0).has_text_changed == 1);

  free_line_text(output.internal_lines_map.at(1));
}
