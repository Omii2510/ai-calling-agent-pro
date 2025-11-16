from flask import Flask, request, Response, render_template, jsonify
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from groq import Groq
from gtts import gTTS
from pymongo import MongoClient
import requests
import os
import datetime

app = Flask(__name__)

# ------------------------- ENVIRONMENT VARIABLES -------------------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

client_twilio = Client(TWILIO_SID, TWILIO_TOKEN)
client_groq = Groq()
mongo = MongoClient(MONGO_URI)
db = mongo["ai_agent"]
calls_collection = db["calls"]


# --------------------------- UI DASHBOARD -------------------------------
@app.route("/")
def home():
    all_calls = list(calls_collection.find().sort("_id", -1))
    return render_template("index.html", calls=all_calls)


# -------------------------- START OUTBOUND CALL -------------------------
@app.route("/call", methods=["POST"])
def call_user():
    target_number = request.json.get("phone")
    webhook = request.json.get("webhook")

    call = client_twilio.calls.create(
        to=target_number,
        from_=TWILIO_NUMBER,
        url=f"{webhook}/voice"
    )

    calls_collection.insert_one({
        "call_sid": call.sid,
        "phone": target_number,
        "start_time": datetime.datetime.utcnow(),
        "messages": []
    })

    return jsonify({"message": "Call started", "call_sid": call.sid})


# --------------------------- /VOICE -------------------------------------
@app.route("/voice", methods=["POST"])
def voice_start():
    response = VoiceResponse()
    response.say(
        "Hello, this is your AI calling agent from AiKing Solutions. May I confirm if you are available for a quick discussion?",
        voice="alice"
    )
    response.record(
        timeout=2,
        maxLength=15,
        playBeep=True,
        action="/recording"
    )
    return Response(str(response), mimetype="text/xml")


# --------------------------- /RECORDING ---------------------------------
@app.route("/recording", methods=["POST"])
def recording_handler():
    call_sid = request.form.get("CallSid")
    recording_url = request.form.get("RecordingUrl") + ".wav"

    # Download audio
    audio_file = requests.get(recording_url, auth=(TWILIO_SID, TWILIO_TOKEN))
    path = f"hr_{call_sid}.wav"
    with open(path, "wb") as f:
        f.write(audio_file.content)

    # STT -----------------------
    with open(path, "rb") as audio:
        stt = client_groq.audio.transcriptions.create(
            file=audio,
            model="whisper-large-v3"
        ).text

    # AI reply ------------------
    prompt = f"""
    You are an AI calling agent. Reply politely, professionally.
    HR said: "{stt}"
    Your response:
    """
    llm_response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    ai_reply = llm_response.choices[0].message.content

    # TTS -----------------------
    tts_path = f"reply_{call_sid}.mp3"
    gTTS(ai_reply, lang="en").save(tts_path)

    # Save to DB ---------------
    calls_collection.update_one(
        {"call_sid": call_sid},
        {"$push": {"messages": {"hr": stt, "agent": ai_reply}}}
    )

    # Play reply ----------------
    response = VoiceResponse()
    public_url = request.url_root.strip("/")
    audio_url = f"{public_url}/{tts_path}"
    response.play(audio_url)
    response.record(timeout=2, maxLength=10, playBeep=True, action="/recording")
    return Response(str(response), mimetype="text/xml")


# -------------------------- SERVE AUDIO FILES ---------------------------
@app.route("/<filename>")
def serve_audio(filename):
    return app.send_static_file(filename)


# -------------------------- CALL ANALYSIS ------------------------------
@app.route("/analysis/<call_sid>")
def analyze(call_sid):
    call_data = calls_collection.find_one({"call_sid": call_sid})
    return jsonify(call_data)


# ---------------------------- MAIN RUN --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
