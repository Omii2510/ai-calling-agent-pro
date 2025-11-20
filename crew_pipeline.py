import os

os.environ["CREWAI_DISABLE_OPENAI"] = "true"
os.environ["CREWAI_DEFAULT_LLM"] = "Groq"
os.environ["CREWAI_USE_GROQ"] = "true"
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE_URL"] = ""

from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

# Disable OpenAI completely
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE_URL"] = ""

# Groq LLM Wrapper
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

# ------------------ AGENTS ------------------

# Agent 1: Understand HR message
hr_understanding_agent = Agent(
    role="HR Understanding Agent",
    goal="Interpret the meaning of the HR's response.",
    backstory="Expert in analyzing HR conversations and extracting context.",
    llm=llm
)

# Agent 2: Generate final reply
reply_agent = Agent(
    role="Reply Generator Agent",
    goal="Generate polite follow-up replies.",
    backstory="Professional corporate assistant.",
    llm=llm
)

# ------------------ CREW PIPELINE ------------------

def run_crew(hr_text):

    # Task 1
    task1 = Task(
        description=f"""
        Interpret this HR message clearly:
        "{hr_text}"
        Extract intent, job details and meaning.
        """,
        agent=hr_understanding_agent,
        expected_output="Short, clear interpretation of HR message."
    )

    # Task 2
    task2 = Task(
        description="""
        Based on the HR interpretation, generate:
        - A polite reply
        - A follow-up question
        - Natural, spoken English style
        """,
        agent=reply_agent,
        expected_output="A short natural conversational reply for phone call."
    )

    crew = Crew(
        agents=[hr_understanding_agent, reply_agent],
        tasks=[task1, task2]
    )

    result = crew.kickoff()
    return result
