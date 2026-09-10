"""Non-blocking macOS text-to-speech speaker."""
import subprocess
import time
import config

class VoiceAssistant:
    def __init__(self):
        self.last_speech_time = 0

    def speak(self, text, force=False):
        now = time.time()
        if force or (now - self.last_speech_time > config.SPEECH_COOLDOWN_SEC):
            subprocess.Popen(["say", text])
            self.last_speech_time = now
