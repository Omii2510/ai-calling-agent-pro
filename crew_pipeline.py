import os
from crewai import Agent, Task, Crew
from groq import Groq

# Disable OpenAI fallback completely
os.environ["OPENAI_API_KEY"] = ""
os.environ["CREWAI_NATIVE_LLM"] = "disabled"
os.environ["CREWAI_ALLOW_FALLBACK"] = "false"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_crew(hr_text):
    """Process HR response, generate AI reply using Groq, CrewAI does flow ONLY."""

    # ------- Dummy CrewAI (no LLM logic) ---------
    interpreter_agent = Agent(
        role="HR Interpreter",
        goal="Coordinate job conversation flow.",
        backstory="Agent without LLM.",
        llm=None,
        allow_delegation=False,
        max_iter=1,
        use_executor=False
    )

    task = Task(
        description="Handle HR reply logically.",
        expected_output="Done.",
        agent=interpreter_agent
    )

    crew = Crew(
        agents=[interpreter_agent],
        tasks=[task],
        verbose=False
    )

    crew.run()  # safe (does NOT use LLM)

    # ------- Actual Groq LLM for Conversation -------
    prompt = (
        "You are an AI calling agent speaking politely to HR. "
        "Analyze their message and continue the job inquiry conversation. "
        "Keep the reply short, human-like, and natural.\n\n"
        f"HR said: {hr_text}"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": hr_text},
        ],
    )

    ai_reply = response.choices[0].message.content.strip()
    return ai_reply
