import os

from google.adk.agents import Agent

agent = Agent(
    name="csm_agent",
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-live-001"),
    instruction="You are a helpful voice assistant. Keep responses concise and conversational.",
)
