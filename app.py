from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime

# ------------------------------------------------
#  IMPORT CrewAI PIPELINE (NO LLM MODE)
# ------------------------------------------------
from crew_pipeline import run_crew

app = Flask(__name__)

# ------------------------------------------------
#  ENV VARIABLES
# ------------------------------------------------
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TARGET_NUMBER = os.environ.get("TARGET_PHONE_NUMBER")

GROQ_KEY = os.environ.get("GROQ_API_KEY")
PUBLIC_URL = os.environ.get("PUBLIC_URL")

# ------------------------------------------------
#  DISABLE OPENAI COMPLETELY (IMPORTANT)
# ------------------------------------------------
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE_URL"] = ""
os.environ["CREWAI_NATIVE_LLM"] = "disabled"
os.environ["CREWAI_ALLOW_FALLBACK"] = "false"

# ------------------------------------------------
#  MongoDB Setup
# ------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI or "mongodb+srv" not in MONGO_URI:
    raise Exception("❌ MONGO_URI missing or invalid! Must be MongoDB Atlas URI.")

mongo = MongoClient(MONGO_URI)
db = mongo["ai-calling-agent"]
calls_collection = db["calls"]

# ------------------------------------------------
#  Twilio + Groq clients
# ------------------------------------------------
twilio_client = Client(TW_SID, TW_TOKEN)
groq_client = Groq(api_key=GROQ_KEY)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")


# ------------------------------------------------
#  HOME PAGE
# ------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------
#  MAKE OUTBOUND CALL
# ------------------------------------------------
@app.route("/call")
def call():
    if not PUBLIC_URL:
        return {"error": "PUBLIC_URL not set"}, 500

    call = twilio_client.calls.create(
        to=TARGET_NUMBER,
        from_=TW_NUMBER,
        url=f"{PUBLIC_URL}/voice"
    )

    return jsonify({"message": "Call started", "call_sid": call.sid})


# ------------------------------------------------
#  AI OPENS THE CALL
# ------------------------------------------------
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()

    resp.say(
        "Hello, this is the AI calling agent from AiKing Solutions. "
        "May I know what job openings are currently available?",
        voice="alice"
    )

    resp.record(
        action="/recording",
        method="POST",
        playBeep=True,
        maxLength=10,
        timeout=2
    )

    return Response(str(resp), mimetype="text/xml")


# ------------------------------------------------
#  HANDLE HR REPLY, AI RESPONDS
# ------------------------------------------------
@app.route("/recording", methods=["POST"])
def recording():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            raise Exception("No RecordingUrl from Twilio!")

        # Download audio file
        hr_audio = "static/hr.wav"
        r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN))
        with open(hr_audio, "wb") as f:
            f.write(r.content)

        # Speech-to-text (Groq Whisper)
        with open(hr_audio, "rb") as audio:
            hr_text = groq_client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3"
            ).text

        print("\nHR SAID:", hr_text)

        # --------------------------------------------------------
        #  CrewAI workflow (no LLM) + LLM Reply (Groq)
        # --------------------------------------------------------
        ai_response = run_crew(hr_text)
        print("AI Response:", ai_response)

        # save in DB
        calls_collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "hr_message": hr_text,
            "ai_message": ai_response,
            "recording_url": recording_url
        })

        # Convert AI text -> speech
        reply_path = "static/ai_reply.mp3"
        gTTS(ai_response, lang="en").save(reply_path)

        resp = VoiceResponse()
        resp.play(f"{PUBLIC_URL}/static/ai_reply.mp3")

        # Listen for next reply
        resp.record(
            action="/recording",
            maxLength=10,
            playBeep=True,
            timeout=2
        )

        return Response(str(resp), mimetype="text/xml")

    except Exception as e:
        print("❌ ERROR in /recording:", e)
        return "Error", 500


# ------------------------------------------------
#  STATIC FILES ROUTE
# ------------------------------------------------
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ------------------------------------------------
#  SUMMARY PAGE
# ------------------------------------------------
@app.route("/summary")
def summary():
    data = list(calls_collection.find().sort("timestamp", -1))
    return render_template("summary.html", calls=data)


# ------------------------------------------------
#  RUN APP
# ------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
