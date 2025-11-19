# -------------- MUST COME FIRST --------------
import os
from dotenv import load_dotenv

# Keep OpenAI disabled; do not set fake endpoints
os.environ["CREWAI_DISABLE_OPENAI"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

load_dotenv()

from crewai import Agent, Task, Crew, LLM

# Use CrewAI's LLM wrapper to bind Groq, preventing OpenAI fallback
groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

def run_crew(hr_text: str) -> str:
    assistant = Agent(
        role="HR Assistant",
        goal="Reply politely in 1–2 sentences.",
        backstory="You help during HR calls.",
        llm=groq_llm,
        verbose=False,
        allow_delegation=False,
        tools=[]
    )

    task = Task(
        description=f"HR said: '{hr_text}'. Respond politely.",
        expected_output="1–2 sentence polite reply",
        agent=assistant
    )

    crew = Crew(
        agents=[assistant],
        tasks=[task],
        llm=groq_llm,
        verbose=False
    )

    result = crew.kickoff()
    try:
        return result.raw_output.strip()
    except Exception:
        # Fall back gracefully if raw_output isn't present
        return str(result).strip()
