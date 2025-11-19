# -------------- MUST COME FIRST --------------
import os

os.environ["CREWAI_DISABLE_OPENAI"] = "true"
os.environ["OPENAI_API_KEY"] = "disabled"
os.environ["OPENAI_BASE_URL"] = "https://api.fake-openai.com/v1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# --------------------------------------------
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# -------------- INIT GROQ LLM ONLY --------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)

def run_crew(hr_text: str) -> str:

    assistant = Agent(
        role="HR Assistant",
        goal="Reply politely in 1–2 sentences.",
        backstory="You help during HR calls.",
        llm=llm,
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
        verbose=False
    )

    result = crew.kickoff()

    try:
        return result.raw_output.strip()
    except:
        return str(result).strip()
