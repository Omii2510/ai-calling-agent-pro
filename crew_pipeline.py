import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# -------------- FORCE DISABLE ALL OPENAI --------------
os.environ["OPENAI_API_KEY"] = "disabled"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["CREWAI_DISABLE_OPENAI"] = "true"
os.environ["OPENAI_BASE_URL"] = "https://invalid-url.com"

# -------------- INIT GROQ LLM --------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# -------------- MAIN FUNCTION --------------
def run_crew(hr_text: str) -> str:
    """
    Use CrewAI + Groq to generate short HR replies.
    """

    assistant = Agent(
        role="HR Call Assistant",
        goal="Reply politely to HR in 1–2 sentences.",
        backstory="You are an AI assistant helping a candidate during HR phone calls.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
        tools=[]  # IMPORTANT → prevents any OpenAI tools from loading
    )

    task = Task(
        description=f"HR said: '{hr_text}'. Give a short polite reply.",
        expected_output="A short, polite, friendly 1–2 sentence reply.",
        agent=assistant
    )

    crew = Crew(
        agents=[assistant],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    # Safe return
    try:
        return result.raw_output.strip()
    except:
        return str(result).strip()
