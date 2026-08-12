#include "transcriber.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("transcript revision survives flag clearing and tracks updates") {
  TranscriptStreamOutput output;

  TranscriberLine first;
  first.id = 7;
  first.just_updated = true;
  first.text = new std::string("one");
  output.ordered_internal_line_ids.push_back(first.id);
  output.add_or_update_line(first);
  output.update_transcript_from_lines();
  REQUIRE(output.transcript.revision == 1);
  REQUIRE(output.transcript.lines[0].has_text_changed == true);

  output.clear_update_flags();
  output.update_transcript_from_lines();
  REQUIRE(output.transcript.revision == 1);

  TranscriberLine second;
  second.id = 7;
  second.just_updated = true;
  second.text = new std::string("two");
  output.add_or_update_line(second);
  output.update_transcript_from_lines();
  REQUIRE(output.transcript.revision == 2);
  REQUIRE(output.transcript.lines[0].has_text_changed == true);
}
