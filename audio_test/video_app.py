import speech_recognition as sr
import pygame
import time
import os 
import logging
from colorama import Fore, Style, init
import warnings
from openai import OpenAI
from groq import Groq
import deepgram
import cv2
import subprocess
from deepgram import DeepgramClient , SpeakOptions
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
# import os
# os.environ["GROQ_API_KEY"] = "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"

GROQ_LLM = ChatGroq(
            groq_api_key="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",
            model="llama3-70b-8192"
            
        )


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

cap = cv2.VideoCapture(0)

def capture_image():
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the image
        image_path = "captured_image.png"
        cv2.imwrite(image_path, frame)
        print("Image captured and saved as captured_image.png")
        return image_path
    return None
  
def record_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source: 
        # logging. info("Recording started")
        audio_data = recognizer.listen(source) # Listen for the first phrase and extract it inte
        # logging. info("Recording complete")
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data.get_wav_data()) # Write the recorded audio data to a WA\

def transcribe_audio(client_1,audio_file_path):

    with open(audio_file_path, "rb") as audio_file:
        transcription = client_1.audio.transcriptions.create(
            model = "whisper-1",
            file=audio_file
        )

    return transcription.text

def generate_response(client, SpeakOptions):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=SpeakOptions
    )
    return response.choices[0].message.content


#########################################################
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a situation Categorizer Agent You are a master at understanding what a user ask when they ask question and are able to categorize it in a useful way

     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Conduct a comprehensive analysis of the user massage provided and categorize into one of the following categories:
        visual - used when someone is asking for  about How he/she Looks or how was environment behind user or any related user visual question about user \
        regular - used when someone is ask about information of non visual question or ask about some topic or queary\
            Output a single cetgory only from the types ('visual', 'regular') \
            eg:
            'visual' \

    User question:\n\n {initial_msg} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_msg"],
)
category_generator = prompt | GROQ_LLM | StrOutputParser()

############################################################################
prompt_01 = PromptTemplate(
    template="""system
    You are an assistant capable of generating detailed answers based on user image descriptions and Make response like you explain user about his situation. in under 40 words

    user
    Here is the description of the image: {image_description}

    Based on this description, provide an answer to the following question: {user_question}

    assistant
    """,
    input_variables=["image_description", "user_question"]
)
visual_generator = prompt_01 | GROQ_LLM | StrOutputParser()

#############################################################################

def text_to_speech(client_1, text,output_file_path):

    speech_response = client_1.audio.speech.create(
        model = "tts-1",
        voice = "fable",
        input = text
    )
    speech_response.stream_to_file(output_file_path)

# def text_to_speech(deepgram,options, response_text,output_file_path):
#     deepgram = DeepgramClient(api_key="aa55e22eb75050440bdeb913bc4586c88a991549")
#     options = SpeakOptions(
#         model="aura-angus-en", # Change voice if needed
#         encoding="linear16",
#         container="wav"
#     )
#     SPEAK_OPTIONS = {"text": response_text}
#     response = deepgram.speak.v("1").save(output_file_path, SPEAK_OPTIONS, options)

def play_audio(file_path):

    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(3)
    pygame.mixer.quit()

# print('kkkkkkkkkkd')
def main():
    client = Groq(api_key="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")
    client_1 = OpenAI(api_key="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")

    chat_history = [
        {"role":"system","content":"you are a helpful assistant keep your answer short and confident"}
    ]

    while True:
        try:

             # Capture image after recording audio
            image_path = capture_image()
            print('speakkkk')
            record_audio('test01.wav')

            user_input = transcribe_audio(client_1,'test01.wav')
            print('user  <><>',user_input)

            chat_history.append({"role":"user","content":user_input})

            visual_regular =  category_generator.invoke({"initial_msg": user_input})
            print('>>',visual_regular)

            if visual_regular == 'visual':
                cmd = "ollama"
                args = ["run", "llava","what is in the image?", "E:/practice/talk_assistant/captured_image.png"]
                message = subprocess.check_output([cmd] + args).decode('utf-8').splitlines()
                print('888',message[1:2],'7777')
                ai_input = visual_generator.invoke({"image_description": message[1:2],"user_question":user_input})
                chat_history.append({"role":"assistant","content":ai_input})
                text_to_speech(client_1, ai_input,'output02.mp3')
                play_audio('output02.mp3')

            else:
                response_text = generate_response(client, chat_history)

                chat_history.append({"role":"assistant","content":response_text})
                print('chat  ',chat_history)

                #time.sleep(1)
                #text_to_speech(deepgram,options, response_text,'output.wav')
                text_to_speech(client_1, response_text,'output01.mp3')
                print('respooo  ',response_text)

                play_audio('output01.mp3')

        except Exception as e:
            print(e)
main()
