from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime

# Import CrewAI pipeline
from crew_pipeline import run_crew

app = Flask(__name__)

# ---------------- ENV VARIABLES ----------------
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TARGET_NUMBER = os.environ.get("TARGET_PHONE_NUMBER")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
PUBLIC_URL = os.environ.get("PUBLIC_URL")

# Critical MongoDB Atlas check
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI or "mongodb+srv" not in MONGO_URI:
    raise Exception("❌ MONGO_URI is missing or NOT Atlas URI!")

mongo = MongoClient(MONGO_URI)
db = mongo["ai-calling-agent"]
calls_collection = db["calls"]

client = Client(TW_SID, TW_TOKEN)
groq = Groq(api_key=GROQ_KEY)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")


@app.route("/")
def home():
    return render_template("index.html")


# ---------------- Make Call ----------------
@app.route("/call")
def call():
    if not PUBLIC_URL:
        return {"error": "PUBLIC_URL not set"}, 500

    call = client.calls.create(
        to=TARGET_NUMBER,
        from_=TW_NUMBER,
        url=f"{PUBLIC_URL}/voice"
    )
    return jsonify({"message": "Call started", "call_sid": call.sid})


# ---------------- Start Voice Flow ----------------
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()

    resp.say(
        "Hello, this is the AI Calling Agent from AiKing Solutions. "
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


# ---------------- Recording Handler ----------------
@app.route("/recording", methods=["POST"])
def recording():
    try:
        recording_url = request.form.get("RecordingUrl") + ".wav"

        # Download HR audio
        hr_audio = "static/hr.wav"
        r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN))
        with open(hr_audio, "wb") as f:
            f.write(r.content)

        # Speech-to-Text (Groq Whisper)
        with open(hr_audio, "rb") as audio:
            hr_text = groq.audio.transcriptions.create(
                file=audio, model="whisper-large-v3"
            ).text

        print("HR Said:", hr_text)

        # ---------------- CrewAI multi-agent response ----------------
        ai_response = run_crew(hr_text)
        print("Crew AI Response:", ai_response)

        # Save to MongoDB
        calls_collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "hr_message": hr_text,
            "ai_message": ai_response,
            "recording_url": recording_url
        })

        # Convert AI response to speech
        reply_path = "static/ai_reply.mp3"
        gTTS(ai_response, lang="en").save(reply_path)

        # Play response + continue conversation
        resp = VoiceResponse()
        resp.play(f"{PUBLIC_URL}/static/ai_reply.mp3")
        resp.record(
            maxLength=10,
            playBeep=True,
            timeout=2,
            action="/recording"
        )

        return Response(str(resp), mimetype="text/xml")

    except Exception as e:
        print("ERROR:", e)
        return "Error", 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/summary")
def summary():
    data = list(calls_collection.find().sort("timestamp", -1))
    return render_template("summary.html", calls=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
