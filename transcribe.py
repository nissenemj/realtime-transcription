import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np

# Lataa äänitiedosto
audio_input, sample_rate = sf.read('audio.wav')

# Lataa prosessori ja malli
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Siirrä malli GPU:lle, jos se on käytettävissä
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Aseta kieli ja tehtävä
language = "fi"  # Vaihda tämä haluamaksesi kieleksi
task = "transcribe"

forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

# Pilko äänitiedosto 30 sekunnin paloihin
chunk_duration = 30  # sekuntia
sample_rate = 16000  # Varmista, että tämä vastaa äänitiedoston näytteenottotaajuutta
chunk_size = chunk_duration * sample_rate

total_length = len(audio_input)
num_chunks = int(np.ceil(total_length / chunk_size))

transcription = ""

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, total_length)
    audio_chunk = audio_input[start:end]

    # Esikäsittele äänidata
    input_features = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt").input_features

    # Generoi transkriptio
    generated_ids = model.generate(
        input_features.to(device),
        forced_decoder_ids=forced_decoder_ids
    )

    # Dekoodaa transkriptio
    chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Lisää transkriptio kokonaistranskriptioon
    transcription += chunk_transcription + " "

    print(f"Osio {i+1}/{num_chunks} transkriboitu.")

# Tallenna transkriptio tekstitiedostoon
with open('transcription.txt', 'w', encoding='utf-8') as f:
    f.write(transcription)

print("Transkriptio valmis. Tulokset on tallennettu tiedostoon 'transcription.txt'.")
