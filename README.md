This is skin disease detection project for use in healthcare systems

<!---
anafkhan03/anafkhan03 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
!pip install -q -U transformers==4.37.2
!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
!pip install -q git+https://github.com/openai/whisper.git
!pip install -q gradio
!pip install -q gTTS
import torch
from transformers import BitsAndBytesConfig, pipeline

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline(
    "image-to-text",
    model=model_id,
    model_kwargs={"quantization_config": quant_config}
    )
    import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS
from PIL import Image
image_path = "/content/download.png"
image = Image.open((image_path))
image
![image](https://github.com/user-attachments/assets/105903bf-1a33-47f4-b1f2-a6a3122d9e48)

import nltk
nltk.download('punkt')
from nltk import sent_tokenize
import locale
print(locale.getlocale())  # Before running the pipeline
# Run the pipeline
print(locale.getlocale())  # After running the pipeline
max_new_tokens = 200

prompt_instructions = """
prompt="User:<image>\n" + prompt_instructions + "\nAssistant:"
outputs=pipe(image,prompt=prompt,generate_kwargs={"max_new_tokens": max_new_tokens})
for sent in sent_tokenize(outputs[0]["generated_text"]):
  print(sent)

  import numpy as np
  torch.cuda.is_available()
  torch.cuda.is_available()
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")
import re
import datetime
## Logger file
tstamp = datetime.datetime.now()
tstamp = str(tstamp).replace(' ','_')
logfile = f'{tstamp}_log.txt'
def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()
import requests
import re
from PIL import Image

def img2txt(input_text, input_image):

    # load the image
    image = Image.open(input_image)

    # Assuming `pipe` and `writehistory` are defined elsewhere
    writehistory(f"Input text: {input_text} - Type: {type(input_text)} - Dir: {dir(input_text)}")
    if isinstance(input_text, tuple):
        prompt_instructions = """
        Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what is the image about?
        """
    else:
        prompt_instructions = """
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
        """ + input_text

    writehistory(f"prompt_instructions: {prompt_instructions}")
    prompt = "USER:<image>\n" + prompt_instructions + "\nAssistant:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})

    if outputs is not None and len(outputs) > 0 and "generated_text" in outputs[0]:
        generated_text = outputs[0]["generated_text"]
        match = re.search(r'Assistant:(.*)', generated_text)
        if match:
            reply = match.group(1).strip()
        else:
            reply = "no response found"
    else:
        reply = "no response generated"

    return reply
    def transcribe(audio):

    # Check if the audio input is None or empty
    if audio is None or audio == '':
        return ('','',None)  # Return empty strings and None audio file

    # language = 'en'

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    return result_text
    def text_to_speech(text, file_path):
    language = 'en'

    audioobj = gTTS(text = text,
                    lang = language,
                    slow = False)

    audioobj.save(file_path)

    return file_path
    import locale
print(locale.getlocale())
import gradio as gr
import base64
import os

# A function to handle audio and image inputs
def process_inputs(audio_path, image_path):
    # Process the audio file (assuming this is handled by a function called 'transcribe')
    speech_to_text_output = transcribe(audio_path)

    # Handle the image input
    if image_path:
        chatgpt_output = img2txt(speech_to_text_output, image_path)
    else:
        chatgpt_output = "No image provided."

    # Assuming 'transcribe' also returns the path to a processed audio file
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")  # Replace with actual path if different

    return speech_to_text_output, chatgpt_output, processed_audio_path

# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Chatgpt Output"),
        gr.Audio("Temp.mp3")
    ],
    title="LLM Powered voice assistant for Multimodal Data",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the interface
iface.launch(debug=True)
