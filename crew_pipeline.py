from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import os

# Correct Groq LLM wrapper
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

# ------------------ AGENTS ------------------

hr_understanding_agent = Agent(
    role="HR Understanding Agent",
    goal="Understand and interpret the HR's response clearly.",
    backstory="Expert in extracting meaning and intent from HR conversations.",
    llm=llm
)

reply_agent = Agent(
    role="Reply Generator Agent",
    goal="Generate polite, natural, professional replies.",
    backstory="A trained corporate conversational assistant.",
    llm=llm
)

# ------------------ CREW PIPELINE ------------------

def run_crew(hr_text):

    task1 = Task(
        description=f"""
            Analyze and interpret this HR message:
            "{hr_text}"
            Focus on meaning and job-related information.
        """,
        agent=hr_understanding_agent,
        expected_output="A short interpretation of HR's message."
    )

    task2 = Task(
        description="""
            Using the HR interpretation, generate:
            • Polite reply
            • Professional tone
            • Natural phone-call style English
            • One simple follow-up question
        """,
        agent=reply_agent,
        expected_output="Final natural AI reply for phone conversation."
    )

    crew = Crew(
        agents=[hr_understanding_agent, reply_agent],
        tasks=[task1, task2]
    )

    # IMPORTANT — New CrewAI function
    result = crew.kickoff()

    return result
