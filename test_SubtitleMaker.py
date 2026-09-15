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


class Word:
    """The parts of a faster-whisper word that split_cue reads."""

    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class Segment:
    """The parts of a faster-whisper segment that build_subtitles reads."""

    def __init__(self, start, end, text, words=None, no_speech_prob=0.0):
        self.start, self.end, self.text = start, end, text
        self.words = words
        self.no_speech_prob = no_speech_prob


def distinct_words(count, length=8):
    """Words of one width that nothing here can mistake for a loop.

    A fixture built by repeating one word is a loop by this app's own
    definition and gets thrown out before the test it was written for ever
    runs. Each of these is a different run of the alphabet, so they share a
    width - which is what the cue arithmetic turns on - without sharing
    letters in the places repetition looks at.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [
        "".join(letters[(index * 7 + step) % 26] for step in range(length))
        for index in range(count)
    ]


def spoken(text, start=0.0, rate=0.25, gaps=()):
    """A segment whose words are laid out end to end at a steady rate.

    Latin words carry their leading space the way faster-whisper returns them.
    `gaps` maps a word index to a silence held before it, which is how a pause
    is put somewhere a cue ought to prefer to break.
    """
    pieces, at = [], start
    for index, unit in enumerate(text.split(" ")):
        at += gaps.get(index, 0.0) if isinstance(gaps, dict) else 0.0
        token = unit if index == 0 else " " + unit
        pieces.append(Word(at, at + rate * len(unit), token))
        at = pieces[-1].end
    joined = "".join(word.word for word in pieces)
    return Segment(start, at, joined.strip(), words=pieces)


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


# Fast enough that eight-character words never reach the ten-second cap, so
# the break tests below measure the break rule rather than that cap.
BRISK = 0.05


class SplitCue(unittest.TestCase):
    """Where a segment is cut into cues, and what time each piece is given.

    Whisper's segment bounds are only where it closed a window. The words are
    where the speech was, so both the cut and the timing come from them.
    """

    def folded(self, cues):
        """How many lines each cue occupies once folded for the screen."""
        return [len(sm.wrap_caption(text).split("\n")) for _, _, text in cues]

    def test_a_segment_that_already_fits_is_left_whole(self):
        cues = sm.split_cue(spoken("a short line of speech"))
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], "a short line of speech")

    def test_a_cue_ends_on_the_last_word_not_on_the_segment(self):
        """A segment closes at the next speech boundary, which can be minutes.

        Timed on the segment, a brief line before a long silence stayed up
        until the ten-second clamp took it down. Timed on its words, it comes
        down when the speaker stopped.
        """
        segment = spoken("a brief line")
        last_word = segment.words[-1].end
        segment.end = last_word + 600.0
        (begins, ends, _), = sm.split_cue(segment)
        self.assertAlmostEqual(begins, segment.words[0].start)
        self.assertAlmostEqual(ends, last_word)

    def test_splits_a_sentence_too_wide_for_one_cue(self):
        segment = spoken(" ".join(distinct_words(12)), rate=BRISK)
        cues = sm.split_cue(segment)
        self.assertGreater(len(cues), 1)
        self.assertLessEqual(max(self.folded(cues)), sm.SUBTITLE_MAX_LINES)

    def test_splits_a_segment_too_long_to_read_at_once(self):
        # Eight short words drawn out: narrow enough to fit on one screen, so
        # only the span can be what forces the split.
        segment = spoken("one two three four five six seven eight", rate=0.5)
        self.assertLessEqual(sm.display_width(segment.text), sm.CUE_MAX_COLUMNS)
        cues = sm.split_cue(segment)
        self.assertGreater(len(cues), 1)
        for begins, ends, _ in cues:
            self.assertLessEqual(ends - begins, sm.MAX_SUBTITLE_DURATION + 1e-9)

    def test_breaks_where_the_speaker_stopped(self):
        """A pause is a better place to end a cue than an even division.

        Both segments here are the same width. With nothing to go on the
        break falls at the halfway mark. A pause before that pulls it there,
        so long as the piece it leaves is still worth having.
        """
        words = " ".join(distinct_words(12))
        even = sm.split_cue(spoken(words, rate=BRISK))
        self.assertEqual(len(even[0][2].split()), 7)

        pulled = sm.split_cue(
            spoken(words, rate=BRISK, gaps={4: sm.CUE_BREAK_PAUSE * 4}))
        self.assertEqual(len(pulled[0][2].split()), 4)

    def test_breaks_where_a_thought_closed(self):
        """Same again, with a full stop doing the pulling instead of a pause."""
        words = distinct_words(12)
        words[3] = words[3][:-1] + "."
        cues = sm.split_cue(spoken(" ".join(words), rate=BRISK))
        self.assertEqual(len(cues[0][2].split()), 4)
        self.assertTrue(cues[0][2].endswith("."))

    def test_ignores_a_break_that_would_leave_the_rest_everything_to_carry(self):
        """A pause in the first few words is not worth cutting at.

        Half an even share is the least a piece may be, so an early comma
        cannot hand the remainder more than it can hold.
        """
        words = distinct_words(12)
        words[0] = words[0][:-1] + ","
        cues = sm.split_cue(spoken(" ".join(words), rate=BRISK))
        self.assertGreater(len(cues[0][2].split()), 1)

    def test_leaves_no_scrap_when_the_speech_never_pauses(self):
        """Continuous speech offers nothing to break on, so it is divided.

        Filling the first cue to the margin instead used to leave a word or
        two stranded at the end - on screen for a fraction of a second, with
        no gap after it to borrow from, because the next cue starts where it
        finishes.
        """
        segment = spoken(" ".join(distinct_words(14)), rate=BRISK)
        cues = sm.split_cue(segment)
        self.assertGreater(len(cues), 1)
        counts = [len(text.split()) for _, _, text in cues]
        self.assertGreaterEqual(min(counts), max(counts) / 2)

    def test_breaks_between_characters_where_there_are_no_spaces(self):
        """Japanese arrives a character at a time and folds at half the count."""
        text = "本日" * 30
        letters = [
            Word(index * 0.4, index * 0.4 + 0.4, char)
            for index, char in enumerate(text)
        ]
        cues = sm.split_cue(Segment(0.0, letters[-1].end, text, words=letters))
        self.assertGreater(len(cues), 1)
        self.assertLessEqual(max(self.folded(cues)), sm.SUBTITLE_MAX_LINES)

    def test_never_loses_or_invents_a_character(self):
        for text in (" ".join(distinct_words(12)),
                     "one two three four five six seven eight",
                     "a short line of speech"):
            segment = spoken(text)
            rebuilt = "".join(
                "".join(piece.split()) for _, _, piece in sm.split_cue(segment)
            )
            self.assertEqual(rebuilt, "".join(text.split()))

    def test_cues_run_forward_and_do_not_overlap(self):
        cues = sm.split_cue(spoken(" ".join(distinct_words(20))))
        for begins, ends, _ in cues:
            self.assertLess(begins, ends)
        for earlier, later in zip(cues, cues[1:]):
            self.assertLessEqual(earlier[1], later[0])

    def test_a_cue_may_be_narrower_than_the_columns_would_allow(self):
        """Words do not divide evenly into lines, so the two budgets differ.

        A cue filled to CUE_MAX_COLUMNS folds onto one line too many whenever
        the last word on a line does not reach the margin exactly. Cutting on
        the fold rather than on the column count is what keeps the promise.
        """
        wide = " ".join(distinct_words(9))
        self.assertLessEqual(sm.display_width(wide), sm.CUE_MAX_COLUMNS)
        self.assertTrue(sm.overflows(wide))
        cues = sm.split_cue(spoken(wide, rate=BRISK))
        self.assertGreater(len(cues), 1)

    def test_a_word_too_wide_to_cut_is_left_for_the_folding(self):
        """A URL has nowhere to break; it must not loop or vanish here."""
        segment = spoken("A" * 200, rate=BRISK)
        cues = sm.split_cue(segment)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], "A" * 200)

    def test_falls_back_to_the_segment_when_there_are_no_words(self):
        """Word timings are asked for, never guaranteed."""
        for words in (None, []):
            segment = Segment(1.0, 4.0, "no timings here", words=words)
            self.assertEqual(sm.split_cue(segment), [(1.0, 4.0, "no timings here")])

    def test_falls_back_when_alignment_did_not_cover_the_segment(self):
        """The transcript matters more than where it would have been cut."""
        segment = spoken("one two three four")
        segment.text = segment.text + " and five"
        cues = sm.split_cue(segment)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], segment.text)


class SilentWindows(unittest.TestCase):
    """Whisper's own reading of a window, used only to lower an existing bar.

    It belongs to the whole window rather than to one cue, which is why it is
    never enough on its own: a line still has to fail the rate test to go.
    """

    SILENT = sm.NO_SPEECH_SUSPECT
    HEARD = sm.NO_SPEECH_SUSPECT - 0.01

    def test_a_silent_window_lowers_the_length_gate(self):
        """The eight-second gate is what lets a shorter invention through."""
        middling = (sm.SUSPECT_MIN_SECONDS + sm.SUSPECT_MIN_SECONDS_UNHEARD) / 2
        # Few enough characters to fail the rate test at this shorter length;
        # a longer line would not, which the next test is about.
        brief = "Bye."
        self.assertLess(len(brief) / middling, sm.HALLUCINATION_MAX_RATE)
        self.assertFalse(sm.is_stretched(brief, middling))
        self.assertTrue(sm.is_stretched(brief, middling, self.SILENT))

    def test_but_is_never_enough_on_its_own(self):
        """A window called silent that nonetheless holds speech keeps it."""
        talking = "we should probably head back before it gets any later"
        self.assertFalse(sm.is_stretched(talking, 12.00, self.SILENT))
        self.assertFalse(sm.is_stretched(talking, 12.00, 1.0))

    def test_and_never_raises_the_gate(self):
        """A window Whisper heard speech in is judged exactly as before."""
        self.assertTrue(sm.is_stretched("Thank you for watching.", 29.98))
        self.assertTrue(
            sm.is_stretched("Thank you for watching.", 29.98, self.HEARD))
        self.assertTrue(sm.is_stretched("Thank you for watching.", 29.98, 0.0))

    def test_below_the_lowered_gate_nothing_is_judged_either_way(self):
        for text in ("はい", "うん", "ん?"):
            self.assertFalse(sm.is_stretched(
                text, sm.SUSPECT_MIN_SECONDS_UNHEARD - 0.01, 1.0))

    def test_a_segment_without_the_reading_is_still_judged(self):
        """Older segments, and the fallbacks, carry no such attribute."""
        bare = Segment(0.0, 29.98, "Thank you for watching.")
        del bare.no_speech_prob
        self.assertEqual(sm.build_subtitles([bare]), [])


class BuildSubtitlesFromWords(unittest.TestCase):
    def wide(self, count=20, start=0.0):
        """A segment wide enough to need several cues and real enough to keep.

        Spoken briskly, so the rate test reads it as speech, and built from
        words that differ, so the loop test does too. A fixture that fails
        either never reaches the splitting this class is about.
        """
        return spoken(" ".join(distinct_words(count)), start=start, rate=BRISK)

    def test_one_segment_can_become_several_cues(self):
        subs = sm.build_subtitles([self.wide()])
        self.assertGreater(len(subs), 1)
        self.assertEqual([s.index for s in subs], list(range(1, len(subs) + 1)))

    def test_every_cue_fits_the_lines_it_is_allowed(self):
        for cue in sm.build_subtitles([self.wide()]):
            self.assertLessEqual(len(cue.content.split("\n")), sm.SUBTITLE_MAX_LINES)

    def test_split_cues_still_never_overlap(self):
        subs = sm.build_subtitles([self.wide(), self.wide(start=60.0)])
        for earlier, later in zip(subs, subs[1:]):
            self.assertLessEqual(earlier.end, later.start)
        for cue in subs:
            self.assertGreater(cue.end, cue.start)

    def test_a_dropped_segment_takes_all_its_cues_with_it(self):
        """The invention tests judge the segment, before anything is cut."""
        invented = spoken("thanks for watching", rate=2.0)
        self.assertGreaterEqual(invented.end - invented.start,
                                sm.SUSPECT_MIN_SECONDS)
        self.assertEqual(sm.build_subtitles([invented]), [])

    def test_output_still_survives_a_real_srt_parser(self):
        subs = sm.build_subtitles([self.wide()])
        self.assertEqual(len(list(srt.parse(srt.compose(subs)))), len(subs))


class ModelCapability(unittest.TestCase):
    def test_reads_english_only_weights_off_the_name(self):
        for name in ("tiny.en", "medium.en", "distil-large-v3.5",
                     "distil-small.en", "nyrahealth/faster_CrisperWhisper"):
            self.assertEqual(sm.model_capability(name), "english_only", name)

    def test_recognises_the_japanese_conversion(self):
        self.assertEqual(
            sm.model_capability("kotoba-tech/kotoba-whisper-v2.0-faster"),
            "japanese_only")

    def test_recognises_weights_that_skipped_the_translate_task(self):
        for name in ("large-v3-turbo", "turbo"):
            self.assertEqual(sm.model_capability(name), "no_translation", name)

    def test_anything_else_is_multilingual(self):
        for name in ("large-v3", "large-v2", "large-v1", "medium", "tiny"):
            self.assertEqual(sm.model_capability(name), "full", name)

    def test_someone_elses_model_is_not_guessed_at(self):
        for name in ("someone/their-whisper", "  someone/their-whisper  "):
            self.assertEqual(sm.model_capability(name), "custom", name)

    def test_every_offered_model_has_a_note_to_show(self):
        for name in sm.MODELS:
            self.assertIn(sm.model_capability(name), sm.MODEL_NOTES, name)

    def test_only_a_model_that_can_translate_is_offered_by_default(self):
        """The dropdown's default must never be the one that warns."""
        self.assertIn(sm.model_capability(sm.DEFAULT_MODEL),
                      sm.CAN_BE_ASKED_TO_TRANSLATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
