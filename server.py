from flask import Flask, request, send_file, jsonify
import os
from TTS.api import TTS
import tempfile

app = Flask(__name__)

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

VOICE_DIR = "voices"
os.makedirs(VOICE_DIR, exist_ok=True)


@app.post("/train-voice")
def train_voice():
    user_id = request.form["user_id"]
    files = request.files.getlist("files")

    user_folder = os.path.join(VOICE_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    for f in files:
        f.save(os.path.join(user_folder, f.filename))

    return {"voice_id": user_id}


@app.post("/tts")
def synthesize():
    data = request.json
    text = data["text"]
    user_id = data["user_id"]

    speaker_folder = os.path.join(VOICE_DIR, user_id)
    speaker_wav = os.listdir(speaker_folder)[0]

    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    tts.tts_to_file(
        text=text,
        speaker_wav=os.path.join(speaker_folder, speaker_wav),
        file_path=out_file
    )

    return send_file(out_file, mimetype="audio/wav")


app.run(host="0.0.0.0", port=10000)
