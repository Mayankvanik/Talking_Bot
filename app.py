import speech_recognition as sr
import pygame
import time
import os 
import logging
from colorama import Fore, Style, init
import warnings
from openai import OpenAI
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = ""

# logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('log_file.log')

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Now, you can use the logger to log messages
logger.info('This is an info message')
logger.debug('This is a debug message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

init(autoreset=True)
  
def record_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source: 
        # logging. info("Recording started")
        audio_data = recognizer.listen(source) # Listen for the first phrase and extract it inte
        # logging. info("Recording complete")
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data.get_wav_data()) # Write the recorded audio data to a WA\

def transcribe_audio(client,audio_file_path):

    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model = "whisper-1",
            file=audio_file
        )

    return transcription.text

def generate_response(client, chat_history):
    messages = [{"role": "system", "content": "you are a helpful assistant keep your answer short and confident"}]
    for message in chat_history:
        if message["role"] == "user":
            messages.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            messages.append({"role": "assistant", "content": message["content"]})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=chat_history
    )
    return response.choices[0].message.content


    # prompt = f"You are a helpful assistant keep your answer short and confident. Question: \n\n {chat_history} \n\n"
    # response = client_1.chat.completions.create(
    #     model="llama3-70b-8192",
    #     prompt=prompt,
    # )
    # return response.choices[0].text

def text_to_speech(client_1, text,output_file_path):

    speech_response = client_1.audio.speech.create(
        model = "tts-1",
        voice = "fable",
        input = text
    )
    speech_response.stream_to_file(output_file_path)

def play_audio(file_path):

    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.quit()

# print('kkkkkkkkkkd')
def main():
    client = OpenAI(api_key="sk-")
    client_1 = ChatGroq(
            model="llama3-70b-8192"
        )

    chat_history = [
        {"role":"system","content":"you are a helpful assistant keep your answer short and confident"}
    ]

    while True:
        try:
            record_audio('test.wav')

            user_input = transcribe_audio(client,'test.wav')
            print('user  ',user_input)

            chat_history.append({"role":"user","content":user_input})

            response_text = generate_response(client, chat_history)

            chat_history.append({"role":"assistant","content":response_text})
            print('chat  ',chat_history)

            text_to_speech(client, response_text,'output.mp3')

            play_audio('output.mp3')

        except Exception as e:
            print(e)

main()