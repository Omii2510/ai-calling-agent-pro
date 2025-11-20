import os
from crewai import Agent, Task, Crew
from crewai.llms import GroqLLM

# Disable OpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["CREWAI_DISABLE_OPENAI"] = "true"

# Use Groq native LLM (CrewAI v1.7+)
llm = GroqLLM(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

hr_understanding_agent = Agent(
    role="HR Understanding Agent",
    goal="Understand and interpret the HR message clearly.",
    backstory="Expert in analyzing job requirements.",
    llm=llm
)

reply_agent = Agent(
    role="Reply Agent",
    goal="Generate polite short replies for HR.",
    backstory="Voice assistant for job calls.",
    llm=llm
)

def run_crew(hr_text):

    task1 = Task(
        description=f"Interpret this HR message: {hr_text}",
        expected_output="Short interpretation of the HR message.",
        agent=hr_understanding_agent
    )

    task2 = Task(
        description="Create a polite reply with one follow-up question.",
        expected_output="Conversational spoken English reply.",
        agent=reply_agent
    )

    crew = Crew(
        agents=[hr_understanding_agent, reply_agent],
        tasks=[task1, task2]
    )

    return crew.kickoff()
