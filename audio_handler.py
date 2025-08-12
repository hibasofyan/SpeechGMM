import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import threading
import wave
from gtts import gTTS
from playsound import playsound


class AudioHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

        self.voices = {}
        self.setup_voices()

        self.sample_rate = 44100
        self.channels = 1
        self.recording = False
        self.audio_buffer = []

    def setup_voices(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            voice_id = voice.id.lower()
            if "hortense" in voice_id and "fr" in voice_id:
                self.voices['Français'] = voice.id
            elif "zira" in voice_id and "en" in voice_id:
                self.voices['Anglais'] = voice.id
            elif "helena" in voice_id and "es" in voice_id:
                self.voices['Espagnol'] = voice.id
            elif "arabic" in voice_id or "ar" in voice_id:
                self.voices['Arabe'] = voice.id
            elif "irina" in voice_id or "ru-ru" in voice_id:
                print("Voix russe détectée :", voice.id)
                self.voices['Russe'] = voice.id

        if 'Russe' not in self.voices:
            self.voices['Russe'] = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_RU-RU_IRINA_11.0'

        default_voice = voices[0].id if voices else None
        for lang in ['Français', 'Anglais', 'Espagnol', 'Arabe', 'Russe']:
            if lang not in self.voices:
                self.voices[lang] = default_voice


    def set_voice_for_language(self, language):
        if language in self.voices:
            self.engine.setProperty('voice', self.voices[language])
            print(f"Voix configurée pour la langue : {language}")
        else:
            print(f"Aucune voix locale trouvée pour {language}, fallback gTTS activé.")
            return False
        return True

    def start_recording(self):
        self.recording = True
        self.audio_buffer = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            if self.recording:
                self.audio_buffer.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            dtype=np.float32
        )
        self.stream.start()

    def stop_recording(self, language_code='fr-FR'):
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)

            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            audio_data = np.int16(audio_data * 32767)

            temp_file = None
            try:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.close()

                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())

                with sr.AudioFile(temp_file.name) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.record(source)

                try:
                    text = self.recognizer.recognize_google(audio, language=language_code)
                    return text
                except sr.UnknownValueError:
                    return "Je n'ai pas compris l'audio"
                except sr.RequestError as e:
                    return f"Erreur de service: {e}"

            finally:
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except PermissionError:
                        pass

        return "Aucun audio enregistré"

    def speak(self, text, language='Français', callback=None):
        def speak_thread():
            local = self.set_voice_for_language(language)
            if local:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                try:
                    lang_code_map = {
                        'Français': 'fr',
                        'Anglais': 'en',
                        'Espagnol': 'es',
                        'Arabe': 'ar',
                        'Russe': 'ru'
                    }
                    lang_code = lang_code_map.get(language, 'en')
                    tts = gTTS(text=text, lang=lang_code)
                    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts.save(temp_path.name)
                    playsound(temp_path.name)
                    os.unlink(temp_path.name)
                except Exception as e:
                    print(f"Erreur gTTS : {e}")
            if callback:
                callback()

        threading.Thread(target=speak_thread).start()

    def get_audio_level(self):
        if self.recording and self.audio_buffer:
            current_buffer = self.audio_buffer[-1]
            rms = np.sqrt(np.mean(current_buffer ** 2))
            return min(1.0, (rms * 15) ** 0.5)
        return 0.0

