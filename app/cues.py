"""One label, two consumers.

The agent writes bracketed cues inline — `[laughter]`, `[sigh]`, `[question-en]`
— and a mood line per reply. This module turns that into three things that must
never disagree:

  * to the engine: the mood picks the reference clip the sentence is spoken in;
  * to the client: the tags and the mood ride alongside the audio frame, so the
    face reacts to the same sentence the ear is hearing;
  * to the transcript: the tags are stripped, so the UI stops rendering `[sigh]`
    as literal text.

Deriving all three from one extraction is the point. A second classifier over
the same sentence would eventually disagree with the first, and a face smiling
through a warning is worse than a face doing nothing.
"""

from __future__ import annotations

import re

from app.tts.base import DEFAULT_MOOD, MOODS, Utterance

# `[laughter]`, `[question-en]`, `[mood:urgent]`. Deliberately narrow: letters,
# digits and hyphens only, so ordinary bracketed prose survives untouched.
_TAG = re.compile(r"\[([a-z][a-z0-9]*(?:[-:][a-z0-9]+)*)\]", re.IGNORECASE)

_MOOD_TAG = re.compile(r"^mood[-:](\w+)$", re.IGNORECASE)

# Cues the face knows how to animate. Anything else is stripped from the
# transcript and passed through to the client, which ignores what it can't use —
# so a new tag is a face change, not a protocol change.
KNOWN_TAGS = frozenset({
    "laughter",
    "sigh",
    "breath",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
})

# When the agent doesn't declare a mood, infer a weak one from the cues it did
# use. Weak on purpose: a wrong strong guess is worse than a neutral one.
_TAG_MOOD = {
    "laughter": "amused",
    "sigh": "concerned",
    "dissatisfaction-hnn": "concerned",
    "surprise-ah": "urgent",
    "surprise-oh": "urgent",
    "surprise-wa": "urgent",
    "surprise-yo": "urgent",
    "confirmation-en": "warm",
}


def extract(text: str, *, index: int = 0, mood: str | None = None) -> Utterance:
    """Split one sentence into what is spoken and what is performed."""
    tags: list[str] = []
    declared = mood

    def take(match: re.Match[str]) -> str:
        nonlocal declared
        tag = match.group(1).lower()
        as_mood = _MOOD_TAG.match(tag)
        if as_mood:
            candidate = as_mood.group(1).lower()
            if candidate in MOODS:
                declared = candidate
            return ""
        tags.append(tag)
        return ""

    spoken = _TAG.sub(take, text)

    # Removing an inline tag leaves a double space, and a tag that stood alone
    # leaves a stray space before its punctuation.
    spoken = re.sub(r"\s{2,}", " ", spoken)
    spoken = re.sub(r"\s+([,.!?;:])", r"\1", spoken).strip()

    if declared not in MOODS:
        declared = next(
            (_TAG_MOOD[t] for t in tags if t in _TAG_MOOD),
            DEFAULT_MOOD,
        )

    return Utterance(
        text=spoken,
        mood=declared,
        tags=tuple(tags),
        index=index,
    )


def strip_tags(text: str) -> str:
    """Tag-free text, for the transcript and for anything that only reads words."""
    return re.sub(r"\s{2,}", " ", _TAG.sub("", text)).strip()
