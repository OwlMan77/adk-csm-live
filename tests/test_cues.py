"""The cue channel: what gets spoken, what gets performed, what gets shown.

These are the three consumers that must never disagree. A face smiling through a
warning is worse than a face doing nothing, and `[sigh]` rendered as literal
text in the transcript is the bug this replaced.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cues import KNOWN_TAGS, extract, strip_tags
from app.tts.base import MOODS


def test_tags_leave_the_spoken_text():
    u = extract("Well [sigh] I suppose we could try again.")
    assert u.text == "Well I suppose we could try again."
    assert u.tags == ("sigh",)
    assert "[" not in u.text


def test_tag_before_punctuation_does_not_leave_a_gap():
    u = extract("Are you sure about that [question-en]?")
    assert u.text == "Are you sure about that?"
    assert u.tags == ("question-en",)


def test_declared_mood_wins_and_is_removed():
    u = extract("[mood:urgent] Get out of the building now!")
    assert u.mood == "urgent"
    assert u.text == "Get out of the building now!"
    assert u.tags == ()


def test_mood_is_inferred_from_cues_when_not_declared():
    assert extract("[laughter] That's a good one.").mood == "amused"
    assert extract("[sigh] If you say so.").mood == "concerned"
    assert extract("Nothing special here.").mood == "neutral"


def test_unknown_mood_falls_back_rather_than_propagating():
    u = extract("[mood:ecstatic] Wonderful news.")
    # An invented mood must not reach the voice: it would miss every reference
    # clip and silently fall through to whatever the default is.
    assert u.mood in MOODS
    assert u.mood == "neutral"


def test_bracketed_prose_and_code_survive():
    # The transcript carries code and citations. Only the cue vocabulary shape
    # is taken, so `[0]` and `[1]` stay where the agent put them.
    u = extract("The array is stored at [0] and [1].")
    assert u.text == "The array is stored at [0] and [1]."
    assert u.tags == ()


def test_multiple_cues_keep_their_order():
    u = extract("[surprise-ah] Oh! I didn't see you there. [laughter]")
    assert u.tags == ("surprise-ah", "laughter")


def test_a_cue_only_sentence_has_nothing_to_speak():
    # "[laughter]" alone must not be synthesised — it would be read aloud.
    u = extract("[laughter]")
    assert u.text == ""
    assert u.tags == ("laughter",)


def test_every_inferred_mood_is_in_the_vocabulary():
    for tag in KNOWN_TAGS:
        assert extract(f"[{tag}] Something.").mood in MOODS


def test_strip_tags_matches_extraction():
    text = "Well [sigh] I suppose so."
    assert strip_tags(text) == extract(text).text
