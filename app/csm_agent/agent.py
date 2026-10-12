import os

from google.adk.agents import Agent

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Cues are stripped from the text before it reaches the speech engine, so they
# are never spoken literally and the vocabulary no longer depends on which
# engine is loaded. They drive the face, and the mood picks which reference clip
# the sentence is cloned from.
INSTRUCTION = (
    "You are a helpful voice assistant. Keep responses concise and conversational.\n"
    "\n"
    "Begin each reply with exactly one mood tag on its own, chosen from: "
    "[mood:neutral], [mood:warm], [mood:urgent], [mood:amused], [mood:concerned]. "
    "Pick the one that fits the whole reply. Use [mood:urgent] only when something "
    "genuinely needs to happen now.\n"
    "\n"
    "Within the reply you may mark non-verbal moments with these tags, written in "
    "lowercase exactly as shown: [laughter], [sigh], [breath], [confirmation-en], "
    "[question-en], [question-ah], [question-oh], [question-ei], [question-yi], "
    "[surprise-ah], [surprise-oh], [surprise-wa], [surprise-yo], "
    "[dissatisfaction-hnn].\n"
    "\n"
    "Place a tag immediately before the words it colours. Use at most one or two "
    "in a reply, and none at all in most replies — they read as insincere when "
    "they are constant. Never invent tags outside this list."
)


def create_agent() -> Agent:
    return Agent(name="csm_agent", model=MODEL, instruction=INSTRUCTION)


# Used by the SSE / text-only endpoints.
agent = Agent(name="csm_agent", model=MODEL, instruction=INSTRUCTION)
