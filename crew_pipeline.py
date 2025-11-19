import os
os.environ["OPENAI_ORGANIZATION"] = ""
os.environ["OPENAI_BASE_URL"] = ""
os.environ["CREWAI_DISABLE_OPENAI"] = "1"


# Initialize Groq LLM
llm = ChatGroq(
api_key=os.getenv("GROQ_API_KEY"),
model="llama-3.1-8b-instant"
)




def run_crew(hr_text: str) -> str:
"""
CrewAI pipeline for generating voice responses from Groq LLM.
Completely free of OpenAI providers or tools.
Returns the AI reply text.
"""


# Agent — tools disabled to avoid OpenAI
assistant_agent = Agent(
role="AI HR Interview Assistant",
goal="Give short, polite, professional replies to HR questions.",
backstory=(
"You help candidates during HR screening calls. "
"You speak clearly and politely. Keep responses short (1-2 sentences)."
),
llm=llm,
verbose=False,
allow_delegation=False,
tools=[] # prevents CrewAI from auto-loading tools that may use OpenAI
)


# Task — CrewAI 1.x requires expected_output
task = Task(
description=f"HR said: '{hr_text}'. Respond politely and naturally.",
expected_output="A friendly, short 1–2 sentence reply.",
agent=assistant_agent
)


# Crew setup
crew = Crew(
agents=[assistant_agent],
tasks=[task],
verbose=False
)


# Run CrewAI
result = crew.kickoff()


# result.raw_output is typically the text output; fall back to str(result) if needed
try:
return result.raw_output
except Exception:
return str(result)
