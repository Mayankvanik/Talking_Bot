import speech_recognition as sr
import pygame
import time
import os 
from colorama import Fore, Style, init
import warnings
from openai import OpenAI
from groq import Groq
import deepgram
import cv2
import subprocess
from deepgram import DeepgramClient, SpeakOptions
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import gradio as gr

# Initialize Gradio and other components
GROQ_LLM = ChatGroq(
    groq_api_key="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",
    model="llama3-70b-8192"
)

init(autoreset=True)

# cap = cv2.VideoCapture(0)
# def capture_image():
#     ret, frame = cap.read()
#     if ret:
#         # Convert the frame from BGR to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # Save the image
#         image_path = "captured_image.png"
#         cv2.imwrite(image_path, frame)
#         print("Image captured and saved as captured_image.png")
#         return image_path
#     return None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
def capture_image():
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB
        # Save the image
        image_path = "captured_image.png"
        cv2.imwrite(image_path, frame)
        print(image_path)
        return image_path
    return None

def record_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source)
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data.get_wav_data())

def transcribe_audio(client_1, audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = client_1.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text

def generate_response(client, chat_history):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=chat_history
    )
    return response.choices[0].message.content


prompt = PromptTemplate(
    template="""system
    You are a situation Categorizer Agent. You are a master at understanding what a user asks when they ask a question and are able to categorize it in a useful way.

    user
    Conduct a comprehensive analysis of the user message provided and categorize it into one of the following categories:
        visual - used when someone is asking about how they look, or how the environment behind them looks, or any related visual question about the user, or tell what is he doing.
        regular - used when someone asks about non-visual information or asks about some topic or query.
    Output a single category only from the types ('visual', 'regular').
    Example:
    'visual'

    User question:\n\n{initial_msg}\n\n

    assistant
    """,
    input_variables=["initial_msg"],
)
category_generator = prompt | GROQ_LLM | StrOutputParser()

prompt_01 = PromptTemplate(
    template="""system
    You are an assistant capable of generating detailed answers based on user image descriptions and making a response like you explain to the user about their situation in under 40 words.

    user
    Here is the description of the image: {image_description}

    Based on this description, provide an answer to the following question: {user_question}

    assistant
    """,
    input_variables=["image_description", "user_question"]
)
visual_generator = prompt_01 | GROQ_LLM | StrOutputParser()

def text_to_speech(client_1, text, output_file_path):
    speech_response = client_1.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text
    )
    speech_response.stream_to_file(output_file_path)

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(3)
    pygame.mixer.quit()

def process_conversation():
    client = Groq(api_key="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")
    client_1 = OpenAI(api_key="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")
    chat_history = [{"role": "system", "content": "you are a helpful assistant, keep your answer short and confident"}]

    try:
        image_path = capture_image()
        print('speakkkk')
        record_audio('test01.wav')

        user_input = transcribe_audio(client_1, 'test01.wav')
        print('user  <><>',user_input)
        chat_history.append({"role": "user", "content": user_input})

        visual_regular = category_generator.invoke({"initial_msg": user_input})
        print('>>',visual_regular)

        if visual_regular == 'visual':
            cmd = "ollama"
            args = ["run", "llava","what is in the image in 20 words?", "E:/practice/talk_assistant/captured_image.png"]
            message = subprocess.check_output([cmd] + args).decode('utf-8').splitlines()
            print('888',message[1:2],'7777')
            ai_input = visual_generator.invoke({"image_description": message[1:2], "user_question": user_input})
            chat_history.append({"role": "assistant", "content": ai_input})
            text_to_speech(client_1, ai_input, 'output_gra.mp3')
            return 'output_gra.mp3'
        else:
            response_text = generate_response(client, chat_history)
            chat_history.append({"role": "assistant", "content": response_text})
            text_to_speech(client_1, response_text, 'output01.mp3')
            return 'output01.mp3'
    except Exception as e:
        print(e)
        return None

def play_audio_gradio(audio_file_path):
    play_audio(audio_file_path)
    return "Audio played successfully."

def show_image():
    image_path = capture_image()
    print('3st')
    return image_path

webcam_output = gr.Video(label="Live Webcam Feed")
def webcam_feed():
    # Code to capture webcam feed goes here
    return webcam_output

app = gr.Blocks()

with app:
    gr.Markdown("# With Visual Talking Bot")
    start_button = gr.Button("Start Conversation")
    output_text = gr.Textbox(label="Audio File Path")
    play_button = gr.Button("Play Audio")
    image_output = gr.Image(label="Captured Image", type="filepath", width=500, height=500)
    #ape = gr.Interface(fn=webcam_feed,inputs=[webcam_output], outputs=None)
    
    start_button.click(fn=process_conversation, inputs=None, outputs= output_text)
    #start_button.click(fn=process_conversation, inputs=None, outputs=[output_text])
    play_button.click(fn=play_audio_gradio, inputs=output_text, outputs=None)
    start_button.click(fn=show_image, inputs=None, outputs=image_output)

if __name__ == "__main__":
    app.launch()
