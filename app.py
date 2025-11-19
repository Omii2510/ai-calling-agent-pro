# app.py
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime
import logging

# CrewAI pipeline
from crew_pipeline import run_crew

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- ENV VARIABLES ----------------
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TARGET_NUMBER = os.environ.get("TARGET_PHONE_NUMBER")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
PUBLIC_URL = os.environ.get("PUBLIC_URL")

# MongoDB Check
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI or "mongodb+srv" not in MONGO_URI:
    raise Exception("❌ MONGO_URI missing or not Atlas URI!")

mongo = MongoClient(MONGO_URI)
db = mongo["ai-calling-agent"]
calls_collection = db["calls"]

# Twilio client (check creds)
if not (TW_SID and TW_TOKEN):
    raise Exception("❌ Twilio credentials missing. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.")

client = Client(TW_SID, TW_TOKEN)

# Groq client instance for audio transcription
if not GROQ_KEY:
    raise Exception("❌ GROQ_API_KEY missing in environment variables.")

groq = Groq(api_key=GROQ_KEY)

# Ensure static exists
if not os.path.exists("static"):
    os.makedirs("static")


@app.route("/")
def home():
    return render_template("index.html")


# ---------------- Make Outgoing Call ----------------
@app.route("/call")
def call():
    if not PUBLIC_URL:
        return {"error": "PUBLIC_URL not set"}, 500

    if not TARGET_NUMBER:
        return {"error": "TARGET_PHONE_NUMBER not set"}, 500

    call = client.calls.create(
        to=TARGET_NUMBER,
        from_=TW_NUMBER,
        url=f"{PUBLIC_URL}/voice"
    )
    return jsonify({"message": "Call started", "call_sid": call.sid})


# ---------------- First Message ----------------
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
        maxLength=12,
        timeout=3
    )

    return Response(str(resp), mimetype="text/xml")


# ---------------- Handle HR Reply ----------------
@app.route("/recording", methods=["POST"])
def recording():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            logging.error("No RecordingUrl received from Twilio")
            return "Bad Request", 400

        # Download audio from Twilio (append .wav if necessary)
        hr_audio = "static/hr.wav"
        # Twilio returns a URL that may require Twilio auth; use requests with auth and a timeout
        r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN), timeout=15)
        r.raise_for_status()
        # === FIXED INDENTATION: ensure the file write is inside the with-block ===
        with open(hr_audio, "wb") as f:
            f.write(r.content)

        # Speech-to-Text (Groq Whisper)
        with open(hr_audio, "rb") as audio_file:
            transcription = groq.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3"
            )

        # robustly extract text
        hr_text = None
        if isinstance(transcription, dict):
            hr_text = transcription.get("text") or transcription.get("result") or ""
        else:
            hr_text = getattr(transcription, "text", None) or ""

        if not hr_text:
            logging.error("Groq transcription returned empty text: %s", transcription)
            hr_text = ""

        logging.info("HR Said: %s", hr_text)

        # ---------------- CrewAI Response ----------------
        ai_response = run_crew(hr_text)
        logging.info("Crew AI Response: %s", ai_response)

        # Save to DB
        calls_collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "hr_message": hr_text,
            "ai_message": ai_response,
            "recording_url": recording_url
        })

        # Text-to-Speech (gTTS saves mp3)
        reply_path = "static/ai_reply.mp3"
        tts = gTTS(ai_response or "Sorry, I couldn't generate a reply.", lang="en")
        tts.save(reply_path)

        # Respond to Twilio: play reply and allow further recording
        resp = VoiceResponse()
        resp.play(f"{PUBLIC_URL}/static/ai_reply.mp3")
        resp.record(
            maxLength=12,
            playBeep=True,
            timeout=3,
            action="/recording"
        )

        return Response(str(resp), mimetype="text/xml")

    except requests.exceptions.RequestException as re:
        logging.exception("Failed to download recording: %s", re)
        return "Error downloading recording", 500
    except Exception as e:
        logging.exception("ERROR in /recording: %s", e)
        return "Error", 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/summary")
def summary():
    data = list(calls_collection.find().sort("timestamp", -1))
    return render_template("summary.html", calls=data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
