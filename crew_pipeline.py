import os
from crewai import Agent, Task, Crew
from crewai.llms import GroqLLM

# Force disable OpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["CREWAI_DISABLE_OPENAI"] = "true"

# Use CrewAI Groq LLM (NOT LangChain)
llm = GroqLLM(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Agents
hr_understanding_agent = Agent(
    role="HR Understanding Agent",
    goal="Understand the HR message clearly.",
    backstory="Expert in job and hiring communication.",
    llm=llm
)

reply_agent = Agent(
    role="Reply Agent",
    goal="Generate polite, short, conversational replies.",
    backstory="Professional AI calling assistant.",
    llm=llm
)

def run_crew(hr_text):

    task1 = Task(
        description=f"Interpret HR message: {hr_text}",
        expected_output="Short, clear interpretation.",
        agent=hr_understanding_agent
    )

    task2 = Task(
        description="Create a polite reply with 1 follow-up question.",
        expected_output="A natural spoken English reply.",
        agent=reply_agent
    )

    crew = Crew(
        agents=[hr_understanding_agent, reply_agent],
        tasks=[task1, task2]
    )

    result = crew.kickoff()
    return result
