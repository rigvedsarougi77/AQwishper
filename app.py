import streamlit as st
import concurrent.futures
import whisper
import librosa
import numpy as np

def process_chunk(model, chunk, file_path):
    audio = chunk
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    transcription = model.transcribe(file_path, fp16=False)['text']
    return transcription

def transcribe_audio(file_path, chunk_size=10):
    model = whisper.load_model('medium')

    # Load the audio file using librosa
    audio, _ = librosa.load(file_path, sr=16000)

    # Process audio in chunks
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

    # Use concurrent processing for faster execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, model, chunk, file_path) for chunk in chunks]

        # Combine transcription results from processed chunks
        transcriptions = [future.result() for future in concurrent.futures.as_completed(futures)]

    return ' '.join(transcriptions)

def main():
    st.title("Audio Transcription App")

    uploaded_file = st.file_uploader("Choose an mp3 audio file", type=["mp3"])

    if uploaded_file is not None:
        # Transcribe audio and display result
        transcription_result = transcribe_audio(uploaded_file)
        st.audio(uploaded_file, format='audio/wav')
        st.write("Transcription:", transcription_result)

if __name__ == "__main__":
    main()
