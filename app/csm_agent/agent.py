import os

from google.adk.agents import Agent

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

INSTRUCTION = "You are a helpful voice assistant. " \
"Keep responses concise and conversational." \
"If you want to denote non verbal and pronunciation cues please use the following [laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], [question-ei], [question-yi], [surprise-ah], [surprise-oh], [surprise-wa], [surprise-yo], [dissatisfaction-hnn]. DO NOT OVERUSE THESE"


def create_agent() -> Agent:
    return Agent(name="csm_agent", model=MODEL, instruction=INSTRUCTION)


# Used by the SSE / text-only endpoints.
agent = Agent(name="csm_agent", model=MODEL, instruction=INSTRUCTION)
