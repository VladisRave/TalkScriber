import os
import json
import uuid
import vosk
import wave
import subprocess


class VoiceRecorder:
    def __init__(self, model_path="./models/vosk-model-small-ru-0.22"):
        self.model = vosk.Model(model_path)
        self.transcript = []

        os.makedirs("./buffer/audio/tmp", exist_ok=True)
        os.makedirs("./buffer/audio/wav", exist_ok=True)
        os.makedirs("./buffer/transcription", exist_ok=True)

    def _save_transcript(self, wav_path, text):
        """
        Сохраняет транскрипт в текстовый файл в папку transcription.
        """
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_path = os.path.join("./buffer/transcription", f"{base_name}.txt")

        with open(output_path, mode="w", encoding="utf-8") as f:
            f.write(text.strip())

        print(f"[INFO] Транскрипция сохранена в {output_path}")

    def save_tmp_and_convert_to_wav(self, tmp_data):
        """
        Сохраняет байты tmp-файла и конвертирует его в WAV (16kHz, mono).
        """
        file_id = str(uuid.uuid4())
        tmp_path = f"./buffer/audio/tmp/{file_id}.tmp"
        wav_path = f"./buffer/audio/wav/{file_id}.wav"

        # Сохраняем .tmp файл
        with open(tmp_path, "wb") as f:
            f.write(tmp_data)

        # Конвертация через ffmpeg
        command = [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            wav_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"[INFO] Сохранён .tmp → {tmp_path}")
        print(f"[INFO] Конвертирован в WAV → {wav_path}")

        return wav_path

    def transcribe_stream(self, wav_path):
        """
        Распознавание WAV-файла по частям.
        """
        wf = wave.open(wav_path, "rb")

        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Неверный формат WAV: нужен 1 канал, 16kHz, PCM 16bit")

        rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
        self.transcript = []

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                self.transcript.append(res.get("text", ""))

        final = json.loads(rec.FinalResult())
        self.transcript.append(final.get("text", ""))

        full_text = " ".join(self.transcript)
        self._save_transcript(wav_path, full_text)

    def transcribe_wav(self, wav_path):
        """
        Обработка WAV-файла целиком.
        """
        wf = wave.open(wav_path, "rb")

        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Неверный формат WAV: нужен 1 канал, 16kHz, PCM 16bit")

        rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
        result_text = ""

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                result_text += result.get("text", "") + " "

        final_result = json.loads(rec.FinalResult())
        result_text += final_result.get("text", "")
        result_text = result_text.strip()

        self._save_transcript(wav_path, result_text)

        return result_text

    def get_full_transcript(self):
        """
        Возвращает полную транскрипцию, объединённую в одну строку.
        """
        return " ".join(self.transcript)

    def clear(self):
        """
        Очищает накопленную транскрипцию.
        """
        self.transcript.clear()