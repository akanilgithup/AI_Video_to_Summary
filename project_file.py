
import streamlit as st
import moviepy
import moviepy as mp
import speech_recognition as sr
import os
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from pydub import AudioSegment



nltk.download('punkt')
nltk.download('stopwords')

import nltk


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')



def extract_audio(video_path, audio_path="temp_audio.wav"):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    return audio_path


def transcribe_audio(file_path, chunk_size=30,lang="en-IN"):
    recognizer = sr.Recognizer()
    print(2)
    audio = AudioSegment.from_wav(file_path)
    duration = len(audio) // 1000  # Convert to seconds

    full_text = []
    for i in range(0, duration, chunk_size):
        chunk = audio[i*1000:(i+chunk_size)*1000]
        chunk.export("temp_chunk.wav", format="wav")
        with sr.AudioFile("temp_chunk.wav") as source:
            audio_data = recognizer.record(source)
            try:
                
                text = recognizer.recognize_google(audio_data, language=lang)
                full_text.append(text)
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except sr.UnknownValueError:
                print("Could not understand audio")

    os.remove("temp_chunk.wav")
    return " ".join(full_text)


# Function to preprocess text for summarization
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
        processed_sentences.append(" ".join(filtered_words))
    return " ".join(processed_sentences)

# Function to generate summary using T5 model
def summarize_text(text):
    try:
        model_name = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = TFT5ForConditionalGeneration.from_pretrained(model_name)

        input_ids = tokenizer.encode("summarize: " + text, return_tensors="tf", max_length=512, truncation=True)
        summary_ids = model.generate(input_ids, max_length=int(len(text) * 0.20), min_length=max(1, int(len(text) * 0.05)), length_penalty=2.0, num_beams=4, early_stopping=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error in summarize_text: {e}")
        return ""

# Function to generate video summary
def generate_summary(video_path):
    audio_path = extract_audio(video_path)
    transcribed_text = transcribe_audio(audio_path)
    
    if transcribed_text:
        processed_text = preprocess_text(transcribed_text)
        summary = summarize_text(processed_text)
        return summary
    else:
        st.error("Failed to transcribe audio")

    if os.path.exists(audio_path):
        os.remove(audio_path)

# Streamlit UI design
def main():
    st.set_page_config(page_title="AI Video Summarizer & Lyrics Generator", page_icon=":clapper:", layout="wide")

    st.title("AI Video Summarizer & Lyrics Generator")
    st.write("Welcome to our application where you can summarize videos and generate summary of text!")

    st.sidebar.title("About")
    st.sidebar.write("This tool uses AI to generate summaries of video content and transcribe text into short summary.")
    st.sidebar.title("Developer :")
    st.sidebar.write("  ANIL KUMAR")
    summary_type = st.sidebar.radio("Select Summary Type", ["Video Summary", "Text Summary"])

    if summary_type == "Text Summary":
        user_input = st.text_area("Enter your text here:")

        if st.button("Summarize"):
            if user_input:
                st.info("Please wait, summarizing...")
                final_summary = summarize_text(preprocess_text(user_input))
                st.header("Summary of Given Text")
                st.success(final_summary)
            else:
                st.warning("Please enter some text.")
    elif summary_type == "Video Summary":
        video_path = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv"])
        if st.button("Generate Summary") and video_path is not None:
            st.warning("Please wait, it may take time for processing...")
            video_file_path = "temp_video.mp4"
            with open(video_file_path, "wb") as f:
                f.write(video_path.getbuffer())
            summary = generate_summary(video_file_path)
            st.text("Summary:")
            st.success(summary)
            if os.path.exists(video_file_path):
                os.remove(video_file_path)

if __name__ == "__main__":
    main()
