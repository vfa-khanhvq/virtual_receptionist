from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# Vietnamese text
text = "Xin chào! Khánh."

# Convert text to speech
tts = gTTS(text=text, lang='vi')
tts_fp = BytesIO()
tts.write_to_fp(tts_fp)
tts_fp.seek(0)

# Load the audio into an AudioSegment
audio = AudioSegment.from_file(tts_fp, format="mp3")

# Play the audio
play(audio)
