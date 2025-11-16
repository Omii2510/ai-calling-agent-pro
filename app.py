from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime
import threading

app = Flask(__name__)

# --------- ENV VARS ----------
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TARGET_NUMBER = os.environ.get("TARGET_PHONE_NUMBER")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
PUBLIC_URL = os.environ.get("PUBLIC_URL")

client = Client(TW_SID, TW_TOKEN)
groq = Groq(api_key=GROQ_KEY)

# MongoDB
mongo = MongoClient(MONGO_URI)
db = mongo["ai_calling_agent"]
calls_collection = db["calls"]

# STATIC
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- START CALL ----------------
@app.route("/call")
def make_call():
    call = client.calls.create(
        to=TARGET_NUMBER,
        from_=TW_NUMBER,
        url=f"{PUBLIC_URL}/voice"
    )
    return jsonify({"call_sid": call.sid, "message": "Call started"})


# ---------------- TWILIO START ----------------
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    resp.say(
        "Hello, this is the AI calling agent from AiKing Solutions. "
        "May I know what job openings are currently available?",
        voice="alice"
    )

    resp.record(
        maxLength=8,
        playBeep=True,
        timeout=1,
        action="/recording",
        method="POST"
    )

    return Response(str(resp), mimetype="text/xml")


# ---------------- BACKGROUND SAVE ----------------
def save_to_mongo(hr_text, ai_text, rec_url):
    try:
        calls_collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "hr_message": hr_text,
            "ai_message": ai_text,
            "recording_url": rec_url
        })
    except Exception as e:
        print("Mongo Error:", e)


# ---------------- RECORDING HANDLER ----------------
@app.route("/recording", methods=["POST"])
def recording():
    recording_url = request.form.get("RecordingUrl") + ".wav"

    # download audio
    audio_path = "static/hr.wav"
    r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN))
    with open(audio_path, "wb") as f:
        f.write(r.content)

    # STT
    with open(audio_path, "rb") as audio:
        hr_text = groq.audio.transcriptions.create(
            file=audio, model="whisper-large-v3"
        ).text

    print("HR:", hr_text)

    # AI
    prompt = f"HR said: {hr_text}. Give polite short response."
    ai_text = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    print("AI:", ai_text)

    # Save in background
    threading.Thread(
        target=save_to_mongo,
        args=(hr_text, ai_text, recording_url)
    ).start()

    # TTS
    tts_file = "static/ai_reply.mp3"
    gTTS(ai_text, lang="en").save(tts_file)

    audio_url = f"{PUBLIC_URL}/static/ai_reply.mp3"

    resp = VoiceResponse()
    resp.play(audio_url)
    resp.record(
        maxLength=8,
        playBeep=True,
        timeout=1,
        action="/recording"
    )

    return Response(str(resp), mimetype="text/xml")


# ---------------- STATIC ----------------
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


# ---------------- SUMMARY ----------------
@app.route("/summary")
def summary():
    try:
        data = list(calls_collection.find().sort("timestamp", -1))
        return render_template("summary.html", calls=data)
    except Exception as e:
        return f"MongoDB Error: {e}", 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
