from flask import Flask, request, jsonify, Response, render_template
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime

app = Flask(__name__)

# ---------------- ENV VARIABLES ----------------
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TARGET_NUMBER = os.environ.get("TARGET_PHONE_NUMBER")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")   # from MongoDB Atlas
PUBLIC_URL = os.environ.get("PUBLIC_URL") # Your Render domain

client = Client(TW_SID, TW_TOKEN)
groq = Groq(api_key=GROQ_KEY)
mongo = MongoClient(MONGO_URI)
db = mongo["ai_calling_agent"]
calls_collection = db["calls"]

# ---------------- HOME UI ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- TRIGGER A CALL ----------------
@app.route("/call")
def make_call():
    call = client.calls.create(
        to=TARGET_NUMBER,
        from_=TW_NUMBER,
        url=f"{PUBLIC_URL}/voice"
    )
    return jsonify({"message": "Call started", "call_sid": call.sid})


# ---------------- TWILIO CALL START ----------------
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()

    resp.say(
        "Hello, this is the AI calling agent from AiKing Solutions. "
        "May I know about the current job openings?",
        voice="alice"
    )

    resp.record(
        maxLength=10,
        playBeep=True,
        timeout=2,
        action="/recording"
    )

    return Response(str(resp), mimetype="text/xml")


# ---------------- RECORDING HANDLER ----------------
@app.route("/recording", methods=["POST"])
def recording():
    recording_url = request.form.get("RecordingUrl") + ".wav"

    # --- DOWNLOAD RECORDING ---
    audio_path = "hr_audio.wav"
    r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN))
    with open(audio_path, "wb") as f:
        f.write(r.content)

    # --- SPEECH TO TEXT ---
    with open(audio_path, "rb") as audio:
        text = groq.audio.transcriptions.create(
            file=audio, model="whisper-large-v3"
        ).text

    print("HR:", text)

    # --- AI RESPONSE ---
    prompt = f"""
    You are an AI calling agent. Respond politely and ask follow-up questions.
    HR said: "{text}"
    Your reply:
    """

    ai_response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    print("AI:", ai_response)

    # --- SAVE TO MONGODB ---
    calls_collection.insert_one({
        "timestamp": datetime.datetime.now(),
        "hr_message": text,
        "ai_message": ai_response,
        "recording_url": recording_url
    })

    # --- TTS ---
    tts_path = "ai_reply.mp3"
    gTTS(ai_response, lang="en").save(tts_path)

    # Serve audio from /static
    audio_url = f"{PUBLIC_URL}/static/ai_reply.mp3"

    resp = VoiceResponse()
    resp.play(audio_url)
    resp.record(maxLength=10, playBeep=True, timeout=2, action="/recording")

    return Response(str(resp), mimetype="text/xml")


# ---------------- SHOW CALL HISTORY ----------------
@app.route("/summary")
def summary():
    data = list(calls_collection.find().sort("timestamp", -1))
    return render_template("summary.html", calls=data)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
