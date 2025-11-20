from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import os

# FORCE CrewAI to NEVER use OpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"]
)

hr_understanding_agent = Agent(
    role="HR Understanding Agent",
    goal="Interpret the HR message.",
    backstory="Expert in job-context interpretation.",
    llm=llm
)

reply_agent = Agent(
    role="Reply Generator Agent",
    goal="Generate polite follow-up reply.",
    backstory="Professional corporate assistant.",
    llm=llm
)

def run_crew(hr_text):
    task1 = Task(
        description=f"Interpret this HR message: {hr_text}",
        agent=hr_understanding_agent
    )

    task2 = Task(
        description="Generate a polite reply with one follow-up question.",
        agent=reply_agent
    )

    crew = Crew(
        agents=[hr_understanding_agent, reply_agent],
        tasks=[task1, task2]
    )

    result = crew.kickoff()
    return result
