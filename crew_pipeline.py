# -------------- MUST COME FIRST --------------
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

# Disable OpenAI completely
os.environ["CREWAI_DISABLE_OPENAI"] = "true"
os.environ["OPENAI_API_KEY"] = "disabled"
os.environ["OPENAI_BASE_URL"] = "https://api.fake-openai.com/v1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# --------------------------------------------
load_dotenv()

# -------------- INIT GROQ LLM ONLY --------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)

def run_crew(hr_text: str) -> str:
    # Define agent with Groq LLM
    assistant = Agent(
        role="HR Assistant",
        goal="Reply politely in 1–2 sentences.",
        backstory="You help during HR calls.",
        llm=llm,                # 👈 Force Groq here
        verbose=True,           # 👈 Turn on for debugging
        allow_delegation=False,
        tools=[]
    )

    # Define task
    task = Task(
        description=f"HR said: '{hr_text}'. Respond politely.",
        expected_output="1–2 sentence polite reply",
        agent=assistant
    )

    # Define crew with Groq LLM
    crew = Crew(
        agents=[assistant],
        tasks=[task],
        llm=llm,                # 👈 Force Groq here too
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()

    try:
        return result.raw_output.strip()
    except Exception:
        return str(result).strip()


# ------------------ TEST ------------------
if __name__ == "__main__":
    reply = run_crew("We need you to join the HR call tomorrow at 10 AM.")
    print("Assistant reply:", reply)
