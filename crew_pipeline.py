# crew_pipeline.py
import os
from crewai import Agent, Task, Crew
from groq import Groq

# Disable OpenAI fully
os.environ["OPENAI_API_KEY"] = ""
os.environ["CREWAI_NATIVE_LLM"] = "disabled"
os.environ["CREWAI_ALLOW_FALLBACK"] = "false"

# -----------------------------------------
# GROQ CLIENT (for generating the AI reply)
# -----------------------------------------
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# -----------------------------------------
# CrewAI AGENTS (NO LLM MODE)
# -----------------------------------------
coordinator = Agent(
    role="Coordinator",
    goal="Coordinate response generation.",
    backstory="You only organize steps, not generate content.",
    llm=None,
    allow_delegation=False,
    max_iter=1
)

processor = Agent(
    role="Processor",
    goal="Process HR input and send it for LLM reply.",
    backstory="You do not generate text yourself.",
    llm=None,
    allow_delegation=False,
    max_iter=1
)

# -----------------------------------------
# Dummy Tasks (DO NOTHING — prevents LLM call)
# -----------------------------------------
task1 = Task(
    description="Organize workflow.",
    expected_output="Done.",
    agent=coordinator
)

task2 = Task(
    description="Prepare HR text for reply.",
    expected_output="Done.",
    agent=processor
)

# -----------------------------------------
# Dummy Crew (NOT executed)
# -----------------------------------------
crew = Crew(
    agents=[coordinator, processor],
    tasks=[task1, task2],
    verbose=False
)

# -----------------------------------------
# MAIN PIPELINE FUNCTION (THIS IS WHAT MATTERS)
# -----------------------------------------
def run_crew(hr_text: str) -> str:
    """
    Simulate CrewAI workflow, but actually generate response using GROQ.
    THIS avoids calling crew.kickoff() — prevents errors.
    """

    prompt = (
        "HR said: '" + hr_text + "'. "
        "Reply politely like an AI HR calling agent, and ask ONE follow-up question."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        reply = response.choices[0].message.content.strip()
        return reply

    except Exception as e:
        print("❌ Groq LLM Error:", e)
        return "Thank you for the information. Could you please tell me more?"
