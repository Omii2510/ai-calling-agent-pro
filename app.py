from flask import Flask, request, jsonify, Response, render_template, send_from_directory
# Download audio from Twilio (append .wav if necessary)
hr_audio = "static/hr.wav"
# Twilio returns a URL that requires the Twilio auth; use requests with auth
r = requests.get(recording_url, auth=(TW_SID, TW_TOKEN), timeout=15)
r.raise_for_status()
with open(hr_audio, "wb") as f:
f.write(r.content)


# Speech-to-Text (Groq Whisper)
with open(hr_audio, "rb") as audio_file:
transcription = groq.audio.transcriptions.create(
file=audio_file,
model="whisper-large-v3"
)


hr_text = getattr(transcription, "text", None) or transcription.get("text")
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
tts = gTTS(ai_response, lang="en")
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
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
