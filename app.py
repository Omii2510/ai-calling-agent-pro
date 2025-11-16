from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime

app = Flask(__name__, static_folder="static")

# ---------------- ENV VARIABLES ----------------
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TARGET_NUMBER = os.environ.get("TARGET_PHONE_NUMBER")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
PUBLIC_URL = os.environ.get("PUBLIC_URL")

# Fail-safe for PUBLIC_URL
if not PUBLIC_URL:
    PUBLIC_URL = "https://ai-calling-agent-pro.onrender.com"

client = Client(TW_SID, TW_TOKEN)
groq = Groq(api_key=GROQ_KEY)

# MongoDB Setup
mongo = MongoClient(MONGO_URI)
db = mongo["ai-calling-agent"]
calls_collection = db["calls"]

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")


# ---------------- HOME UI ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- START CALL ----------------
@app.route("/call", methods=["GET"])
def make_call():
    try:
        call = client.calls.create(
            to=TARGET_NUMBER,
            from_=TW_NUMBER,
            url=f"{PUBLIC_URL}/voice"
        )
        return jsonify({"message": "Call started", "call_sid": call.sid})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- TWILIO CALL START ----------------
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()

    resp.say(
        "Hello, I am the AI calling agent from AiKing Solutions. "
        "May I know what job openings are currently available?",
        voice="alice"
    )

    resp.record(
        maxLength=12,
        playBeep=True,
        timeout=2,
        action="/recording",
        method="POST"
    )

    return Response(str(resp), mimetype="text/xml")


# ---------------- RECORDING HANDLER ----------------
@app.route("/recording", methods=["POST"])
def recording():
    try:
        recording_url = request.form.get("RecordingUrl") + ".wav"

        # --- DOWNLOAD HR AUDIO ---
        hr_audio_path = "static/hr_audio.wav"
        r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN))
        with open(hr_audio_path, "wb") as f:
            f.write(r.content)

        # --- SPEECH TO TEXT ---
        with open(hr_audio_path, "rb") as audio:
            hr_text = groq.audio.transcriptions.create(
                file=audio, model="whisper-large-v3"
            ).text

        print("HR:", hr_text)

        # --- AI RESPONSE ---
        prompt = f"""
        You are an AI calling agent. Always respond politely and professionally.
        HR said: "{hr_text}"
        Your reply:
        """

        ai_response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        print("AI:", ai_response)

        # --- SAVE TO MONGO ---
        calls_collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "hr_message": hr_text,
            "ai_message": ai_response,
            "recording_url": recording_url
        })

        # --- AI TTS Reply ---
        tts_path = "static/ai_reply.mp3"
        gTTS(ai_response, lang="en").save(tts_path)

        audio_url = f"{PUBLIC_URL}/static/ai_reply.mp3"

        # Continue conversation
        resp = VoiceResponse()
        resp.play(audio_url)
        resp.record(
            maxLength=12,
            playBeep=True,
            timeout=2,
            action="/recording"
        )

        return Response(str(resp), mimetype="text/xml")

    except Exception as e:
        print("🔥 Error:", str(e))
        resp = VoiceResponse()
        resp.say("I am sorry, an application error has occurred.", voice="alice")
        return Response(str(resp), mimetype="text/xml")


# ---------------- SUMMARY PAGE ----------------
@app.route("/summary")
def summary():
    try:
        data = list(calls_collection.find().sort("timestamp", -1))
        return render_template("summary.html", calls=data)
    except Exception as e:
        return f"Error loading summary: {e}", 500


# ---------------- SERVE STATIC FILES ----------------
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
