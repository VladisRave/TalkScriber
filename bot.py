import os
import uuid
import json
import shutil
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.utils import executor
from pydub import AudioSegment
from pydub.utils import which
from VoiceRecorder import VoiceRecorder
from TextSummarizer import TextSummarizer


# Загрузка конфигурации из config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ffmpeg_path = config.get("ffmpeg_path")
telegram_token = config.get("telegram_token")

if not ffmpeg_path or not telegram_token:
    raise ValueError("Конфигурация повреждена: отсутствует ffmpeg_path или telegram_token")

# Настройка pydub
AudioSegment.converter = which(ffmpeg_path)

# Логирование
logging.basicConfig(level=logging.INFO)

# Инициализация бота
bot = Bot(token=telegram_token)
dp = Dispatcher(bot)

# Инициализация компонентов
recorder = VoiceRecorder()
summarizer = TextSummarizer()

# Создание буферных директорий
os.makedirs("./buffer/audio/tmp", exist_ok=True)
os.makedirs("./buffer/audio/wav", exist_ok=True)
os.makedirs("./buffer/transcription", exist_ok=True)

# Telegram Меню
start_menu = ReplyKeyboardMarkup(resize_keyboard=True)
start_menu.add(KeyboardButton("🎙Record"), KeyboardButton("⚙️Help"))

main_menu = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
main_menu.add(
    KeyboardButton("📕Markdown"), KeyboardButton("📗LaTeX"),
    KeyboardButton("📘TXT"), KeyboardButton("📙Note")
)
main_menu.add(KeyboardButton("🎙Record"), KeyboardButton("🧹Clear"))


@dp.message_handler(commands=["start", "help"])
async def handle_start(message: types.Message):
    recorder.clear()
    await message.reply("Привет! Давай начнем!", reply_markup=start_menu)


@dp.message_handler(lambda message: message.text == "⚙️Help")
async def handle_help(message: types.Message):
    await message.reply(
        "👋 Привет от команды разработчиков!\n\n"
        "Если у вас возникли вопросы по работе приложения — не стесняйтесь, пишите @Vladis_Rave. Мы всегда на связи ❤️"
    )


@dp.message_handler(lambda message: message.text == "🎙Record")
async def handle_new_session(message: types.Message):
    recorder.clear()
    await message.reply("Запишите голосовое или отправьте аудиофайл", reply_markup=ReplyKeyboardRemove())


@dp.message_handler(content_types=[
    types.ContentType.VOICE,
    types.ContentType.AUDIO,
    types.ContentType.DOCUMENT
])
async def handle_audio_input(message: types.Message):
    recorder.clear()
    file_id = None

    if message.voice:
        file_id = message.voice.file_id
    elif message.audio:
        file_id = message.audio.file_id
    elif message.document:
        file_id = message.document.file_id

    if not file_id:
        await message.answer("‼️Ошибка: не могу определить тип файла😔")
        return

    file = await bot.get_file(file_id)
    file_id = str(uuid.uuid4())
    input_path = f"./buffer/audio/tmp/{file_id}.ogg"
    wav_path = f"./buffer/audio/wav/{file_id}.wav"

    print(f"Получен файл, путь: {input_path}")
    await bot.download_file(file.file_path, input_path)

    if not os.path.exists(input_path):
        await message.answer(f"‼️Ошибка: файл не найден по пути {input_path}.")
        return

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        os.remove(input_path)
    except Exception as e:
        await message.answer(f"‼️Ошибка при конвертации файла: {str(e)}")
        return

    await message.answer("Аудио получено, начинаю распознавание🤔")

    if message.voice:
        recorder.transcribe_stream(wav_path)
    else:
        text = recorder.transcribe_wav(wav_path)
        recorder.transcript = [text]

    os.remove(wav_path)
    await message.answer("Текст получен. Выбери формат конспекта🧑‍💻", reply_markup=main_menu)


@dp.message_handler(lambda message: message.text == "🧹Clear")
async def handle_clear_buffers(message: types.Message):
    for folder in ["./buffer/audio/tmp", "./buffer/audio/wav", "./buffer/transcription"]:
        try:
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            await message.answer(f"‼️Ошибка при очистке {folder}: {str(e)}")
            return
    recorder.clear()
    await message.answer("Буферы очищены🧼")


@dp.message_handler(lambda message: message.text in [
    "📕Markdown", "📗LaTeX", "📘TXT", "📙Note"
])
async def summarize_text(message: types.Message):
    mode = message.text.lower()
    text = recorder.get_full_transcript()
    if not text:
        await message.answer("Нет текста для конспекта🫨")
        return
    await message.answer("Генерирую конспект✍️")
    summary = summarizer.summarize(text, mode=mode)
    await message.answer(summary)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
