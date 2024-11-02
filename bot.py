import asyncio
from telebot.async_telebot import AsyncTeleBot
import os
from face_reco import FaceRecognition
import cv2

API_TOKEN = "8193819933:AAEP5frbpuXH91Ou6yB81SahmPcr1HQuiAs"
DOWNLOAD_PATH = "circles/"
reco = FaceRecognition()

bot = AsyncTeleBot(API_TOKEN)

# States to track the current action
is_verifying = {}
is_uploading = {}

if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

@bot.message_handler(commands=['help', 'start'])
async def send_welcome(message):
    text = "Hi, I'm FaceReco bot. You can use /verify to check a face or /upload_face to add a new face to the dataset."
    await bot.reply_to(message, text)

@bot.message_handler(commands=['verify'])
async def verify_face(message):
    is_verifying[message.chat.id] = True
    is_uploading[message.chat.id] = False
    await bot.reply_to(message, "Please send a video note for verification.")

@bot.message_handler(commands=['upload_face'])
async def upload_face(message):
    is_uploading[message.chat.id] = True
    is_verifying[message.chat.id] = False
    await bot.reply_to(message, "Please send a video note to upload a new face.")

@bot.message_handler(content_types=['video_note'])
async def handle_video_note(message):
    if is_verifying.get(message.chat.id):
        await process_video_note_for_verification(message)
        is_verifying[message.chat.id] = False  # Reset state
    elif is_uploading.get(message.chat.id):
        await process_video_note_for_upload(message)
        is_uploading[message.chat.id] = False  # Reset state
    else:
        await bot.reply_to(message, "Please use /verify or /upload_face first.")

async def process_video_note_for_verification(message):
    file_info = await bot.get_file(message.video_note.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    
    file_name = f"{DOWNLOAD_PATH}{message.video_note.file_id}.mp4"
    
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    cap = cv2.VideoCapture(file_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / 5) for i in range(1, 6)]  

    matches_found = set()

    for frame_number in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        embedding = reco.encode_face(frame)
        
        if embedding is not None:
            match, face_id = reco.is_match(embedding)
            if match:
                print(f"Match found for face ID: {face_id} in frame {frame_number}")
                matches_found.add(face_id)
            else:
                print("No match found in this frame.")

    cap.release()

    if not matches_found:
        await bot.reply_to(message, "Verification not passed, please try again.")
    else:
        await bot.reply_to(message, "Verification passed.")

async def process_video_note_for_upload(message):
    file_info = await bot.get_file(message.video_note.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    
    file_name = f"{DOWNLOAD_PATH}{message.video_note.file_id}.mp4"
    
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    cap = cv2.VideoCapture(file_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / 5) for i in range(1, 6)]  

    embeddings_uploaded = 0

    for frame_number in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        embedding = reco.encode_face(frame)
        
        if embedding is not None:
            reco.store_face(embedding)
            embeddings_uploaded += 1

    cap.release()

    if embeddings_uploaded == 0:
        await bot.reply_to(message, "No face detected, please try again.")
    else:
        await bot.reply_to(message, f"{embeddings_uploaded} faces uploaded successfully.")

@bot.message_handler(func=lambda message: True)
async def echo_message(message):
    await bot.reply_to(message, message.text)


asyncio.run(bot.polling())
