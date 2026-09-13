"""Tests for SubtitleMaker.

Run them with:

    .venv/Scripts/python.exe -m unittest -v        # Windows
    .venv/bin/python -m unittest -v                # Linux, macOS

Nothing here loads a Whisper model or touches the network, so the whole file
runs in a second. What it covers is everything that decides what ends up in an
.srt: how a cue is folded and timed, and which lines are thrown away for having
been invented rather than heard. Those judgements are all thresholds, and a
threshold with no test around it is a threshold that drifts.

Every threshold here was settled by measuring real transcripts, and the tests
say what the measurement was, because a number means nothing without it. The
sample lines themselves are stand-ins written to the same shape - the same
length, the same rate, the same side of the boundary - rather than anything
lifted from a transcript.
"""

import unittest
from datetime import timedelta

import srt

import SubtitleMaker as sm


class Segment:
    """The parts of a faster-whisper segment that build_subtitles reads."""

    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class DisplayWidth(unittest.TestCase):
    def test_counts_east_asian_characters_as_two_columns(self):
        self.assertEqual(sm.display_width("abc"), 3)
        self.assertEqual(sm.display_width("そうだね"), 8)
        # Mixed text adds up as the sum of its parts.
        self.assertEqual(sm.display_width("ok です"), 3 + 4)

    def test_empty(self):
        self.assertEqual(sm.display_width(""), 0)


class WrapCaption(unittest.TestCase):
    def widths(self, wrapped):
        return [sm.display_width(line) for line in wrapped.split("\n")]

    def test_short_lines_are_left_alone(self):
        for text in ("Short line.", "そうだね", ""):
            self.assertEqual(sm.wrap_caption(text), text)

    def test_folds_a_long_latin_sentence(self):
        text = ("There's a lot of expensive cups in rich people's houses, "
                "so be careful not to break one.")
        wrapped = sm.wrap_caption(text)
        self.assertGreater(len(wrapped.split("\n")), 1)
        self.assertLessEqual(max(self.widths(wrapped)), sm.SUBTITLE_LINE_COLUMNS)

    def test_folds_japanese_at_half_the_character_count(self):
        # 56 characters, 112 columns: one line would run off any frame.
        text = ("昨日は朝から雨が降っていたので、予定していた散歩をやめて、"
                "家で本でも読んで過ごすことにしました。")
        wrapped = sm.wrap_caption(text)
        self.assertLessEqual(max(self.widths(wrapped)), sm.SUBTITLE_LINE_COLUMNS)
        self.assertGreaterEqual(len(wrapped.split("\n")), 3)

    def test_lines_are_balanced_not_filled_to_the_brim(self):
        """A cue must not break as one full line and one stray word."""
        text = "a " * 40
        widths = self.widths(sm.wrap_caption(text.strip()))
        self.assertLess(max(widths) - min(widths), sm.SUBTITLE_LINE_COLUMNS / 2)

    def test_never_loses_or_invents_characters(self):
        for text in ("hello world " * 9, "あ" * 60, "a、b。c!d?e" * 9):
            wrapped = sm.wrap_caption(text)
            self.assertEqual("".join(wrapped.split()), "".join(text.split()))

    def test_breaks_a_run_with_nowhere_to_break(self):
        """A URL or long identifier must not run off the side of the frame."""
        wrapped = sm.wrap_caption("A" * 200)
        self.assertLessEqual(max(self.widths(wrapped)), sm.SUBTITLE_LINE_COLUMNS)

    def test_closing_punctuation_is_not_stranded(self):
        # Contrived so a break lands right where the full stop is.
        text = "あ" * 21 + "。" + "い" * 21
        for line in sm.wrap_caption(text).split("\n")[1:]:
            self.assertNotIn(line[0], sm.NEVER_STARTS_A_LINE)

    def test_no_line_is_blank_or_padded(self):
        for text in ("  spaced   out  words  " * 6, "、" * 40):
            for line in sm.wrap_caption(text).split("\n"):
                self.assertTrue(line)
                self.assertEqual(line, line.strip())


class InventedLines(unittest.TestCase):
    """Thresholds measured over 579 cues from three files in two languages."""

    def test_drops_a_phrase_smeared_over_a_whole_window(self):
        # Whisper's stock hallucination, and its Japanese twin.
        self.assertTrue(sm.is_stretched("Thank you for watching.", 29.98))
        self.assertTrue(sm.is_stretched("ご視聴ありがとうございました", 29.98))
        self.assertTrue(sm.is_stretched("すみません", 20.12))

    def test_keeps_real_speech_that_merely_runs_long(self):
        # The slowest genuine line measured ran at 1.5 characters a second, so
        # the threshold sits a third below it. These two sit either side of it.
        self.assertFalse(sm.is_stretched("We should head back soon.", 13.00))
        self.assertFalse(sm.is_stretched(
            "昨日は朝から雨が降っていたので、予定していた散歩をやめて、"
            "家で本でも読んで過ごすことにしました。", 23.50))

    def test_a_short_cue_is_never_judged(self):
        """Real speech is full of brief lines that look wrong by any measure."""
        for text in ("はい", "うん", "ん?"):
            self.assertFalse(sm.is_stretched(text, sm.SUSPECT_MIN_SECONDS - 0.01))
            self.assertFalse(sm.is_looping(text * 9, sm.SUSPECT_MIN_SECONDS - 0.01))

    def test_drops_a_line_repeated_until_the_window_was_full(self):
        for text in ("Ah, " * 50, "フフ" * 25, "I'm sorry, " * 12, "Æ" * 50):
            self.assertTrue(sm.is_looping(text, 29.98), text[:16])

    def test_keeps_someone_talking_fast(self):
        """A word said three times over scores a flat 1.0, same as a loop.

        Only the length tells them apart: this takes under two seconds, and a
        loop runs to the end of the window.
        """
        self.assertEqual(sm.repetition("はいはいはいはい"), 1.0)
        self.assertFalse(sm.is_looping("はいはいはいはい", 1.86))
        self.assertFalse(sm.is_looping("Yes, yes, yes.", 2.00))

    def test_too_few_characters_to_score(self):
        """A word doubled is not evidence of anything."""
        self.assertEqual(sm.repetition("はいはい"), 0.0)

    def test_repetition_stays_in_range_on_awkward_input(self):
        for text in ("", " ", "a", "ab", "x" * 400, "。、！？" * 50):
            self.assertGreaterEqual(sm.repetition(text), 0.0)
            self.assertLessEqual(sm.repetition(text), 1.0)


class DropRepeatRuns(unittest.TestCase):
    def cues(self, *texts):
        return [
            srt.Subtitle(index=0, start=timedelta(seconds=i),
                         end=timedelta(seconds=i + 1), content=text)
            for i, text in enumerate(texts)
        ]

    def test_collapses_a_long_run(self):
        kept = sm.drop_repeat_runs(self.cues("a", "x", "x", "x", "x", "b"))
        self.assertEqual([c.content for c in kept], ["a", "x", "b"])

    def test_leaves_a_pair_alone(self):
        """Someone really can say the same thing twice."""
        kept = sm.drop_repeat_runs(self.cues("a", "x", "x", "b"))
        self.assertEqual([c.content for c in kept], ["a", "x", "x", "b"])


class BuildSubtitles(unittest.TestCase):
    def test_numbers_cues_without_gaps(self):
        subs = sm.build_subtitles([Segment(i, i + 0.5, "line") for i in range(5)])
        self.assertEqual([s.index for s in subs], list(range(1, len(subs) + 1)))

    def test_drops_empty_segments(self):
        self.assertEqual(sm.build_subtitles([Segment(0, 1, "   ")]), [])

    def test_a_brief_cue_is_given_time_to_be_read(self):
        subs = sm.build_subtitles([Segment(0.0, 0.2, "flash"),
                                   Segment(20.0, 21.0, "later")])
        self.assertAlmostEqual(
            (subs[0].end - subs[0].start).total_seconds(), sm.MIN_SUBTITLE_DURATION)

    def test_but_never_at_the_expense_of_the_next_cue(self):
        subs = sm.build_subtitles([Segment(0.0, 0.2, "a"), Segment(0.4, 0.6, "b")])
        self.assertLessEqual(subs[0].end, subs[1].start)

    def test_caps_a_segment_left_open(self):
        # Enough words for the span to be believable; a sparser line this long
        # would be thrown out as invented before the cap ever applied.
        spoken = "and then we walked all the way back down to the harbour again"
        subs = sm.build_subtitles([Segment(0.0, 25.0, spoken)])
        self.assertEqual(len(subs), 1)
        self.assertLessEqual(
            (subs[0].end - subs[0].start).total_seconds(), sm.MAX_SUBTITLE_DURATION)

    def test_cues_never_overlap_and_always_have_duration(self):
        runs = [
            [Segment(0.0, 30.0, "long"), Segment(1.0, 2.0, "starts inside it")],
            [Segment(0.0, 0.2, "a"), Segment(0.3, 0.5, "b"), Segment(0.6, 0.8, "c")],
            [Segment(5.0, 5.0, "zero length")],
        ]
        for segments in runs:
            subs = sm.build_subtitles(segments)
            for cue in subs:
                self.assertGreater(cue.end, cue.start)
            for earlier, later in zip(subs, subs[1:]):
                self.assertLessEqual(earlier.end, later.start)

    def test_output_survives_a_real_srt_parser(self):
        subs = sm.build_subtitles([
            Segment(0.0, 0.2, "flash"),
            Segment(5.0, 9.0, "昨日は朝から雨が降っていたので、"
                              "予定していた散歩をやめました。"),
        ])
        self.assertEqual(len(list(srt.parse(srt.compose(subs)))), len(subs))


class ClipWindows(unittest.TestCase):
    def test_covers_the_whole_timeline_without_gaps(self):
        for duration in (0.5, 29.9, 30.0, 30.1, 95.0, 3600.0):
            clips = sm.split_clips([{"start": 0.0, "end": duration}])
            self.assertAlmostEqual(clips[0]["start"], 0.0)
            self.assertAlmostEqual(clips[-1]["end"], duration)
            for earlier, later in zip(clips, clips[1:]):
                self.assertAlmostEqual(earlier["end"], later["start"])

    def test_no_window_exceeds_what_the_encoder_takes(self):
        """A longer clip is silently truncated by the batched pipeline."""
        for duration in (31.0, 300.0, 5000.0):
            for clip in sm.split_clips([{"start": 0.0, "end": duration}]):
                self.assertLessEqual(
                    clip["end"] - clip["start"], sm.CHUNK_SECONDS + 1e-9)

    def test_divides_evenly_rather_than_leaving_a_sliver(self):
        """A two-second remainder padded out with silence invites invention."""
        spans = [c["end"] - c["start"]
                 for c in sm.split_clips([{"start": 0.0, "end": 32.0}])]
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0], spans[1])


class LanguageSampling(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.audio = np.zeros(int(600 * sm.AUDIO_SAMPLE_RATE), dtype=np.float32)
        self.window = int(sm.CHUNK_SECONDS * sm.AUDIO_SAMPLE_RATE)

    def test_falls_back_to_the_opening_when_no_speech_was_found(self):
        sample = sm.language_montage(self.audio, [])
        self.assertEqual(len(sample), sm.LANGUAGE_SAMPLE_WINDOWS * self.window)

    def test_samples_from_across_the_file(self):
        regions = [{"start": 10.0, "end": 40.0}, {"start": 500.0, "end": 560.0}]
        self.assertGreater(len(sm.language_montage(self.audio, regions)), 0)

    def test_a_region_at_the_very_end_does_not_overrun(self):
        sample = sm.language_montage(self.audio, [{"start": 599.0, "end": 600.0}])
        self.assertGreater(len(sample), 0)
        self.assertLessEqual(len(sample), sm.LANGUAGE_SAMPLE_WINDOWS * self.window)


class OutputPaths(unittest.TestCase):
    def test_writes_beside_the_source(self):
        self.assertEqual(sm.unique_srt_path("/media/video.mp4", set()),
                         "/media/video.srt")

    def test_keeps_both_when_two_sources_would_collide(self):
        taken = {"/media/video.srt"}
        self.assertEqual(sm.unique_srt_path("/media/video.mkv", taken),
                         "/media/video.mkv.srt")


class StatusLine(unittest.TestCase):
    def test_keeps_a_long_name_to_a_fixed_length(self):
        name = "a" * 200 + ".mp4"
        shortened = sm.shorten(name, limit=40)
        self.assertLessEqual(len(shortened), 40)
        self.assertTrue(shortened.endswith(".mp4"))

    def test_leaves_a_short_name_alone(self):
        self.assertEqual(sm.shorten("clip.mp4"), "clip.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
