from crewai import Agent, Task, Crew
from groq import Groq
import os

# Initialize Groq LLM
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ------------------ AGENTS ------------------

# Agent 1 — Understand the HR message
hr_understanding_agent = Agent(
    role="HR Understanding Agent",
    goal="Understand and interpret the HR's response clearly.",
    backstory="You are trained to analyze HR messages, extract meaning, intent, and context.",
    llm=groq_client
)

# Agent 2 — Generate a polite follow-up reply
reply_agent = Agent(
    role="Reply Generator Agent",
    goal="Generate polite, professional replies with follow-up questions.",
    backstory="You sound like a trained corporate assistant from AiKing Solutions.",
    llm=groq_client
)


# ------------------ CREW PIPELINE ------------------

def run_crew(hr_text):
    """
    Multi-agent pipeline to:
    1. Understand the HR message
    2. Generate a polite follow-up AI reply
    """

    # Task 1 — understanding HR
    task1 = Task(
        description=f"""
        Analyze this HR message:
        "{hr_text}"

        Extract meaning, intent, and what HR is trying to convey.
        """,
        agent=hr_understanding_agent,
        expected_output="A clear, short interpretation of HR's message."
    )

    # Task 2 — generate final AI reply
    task2 = Task(
        description="""
        Based on the HR interpretation, generate:

        - A polite reply  
        - Professional language  
        - A follow-up question  
        - Keep it short and natural for a phone call  
        """,
        agent=reply_agent,
        expected_output="A clean, natural, spoken AI response."
    )

    # Create multi-agent crew
    crew = Crew(
        agents=[hr_understanding_agent, reply_agent],
        tasks=[task1, task2]
    )

    # Run pipeline (CrewAI executes tasks automatically)
    result = crew.run()

    return result
