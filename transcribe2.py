import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
from pyannote.core import Segment, Annotation
from pyannote.database.util import load_rttm
import numpy as np
import argparse
import os

# Määritä komentoriviparametrit
parser = argparse.ArgumentParser(description='Transkriptoi äänitiedosto puhujien erottelulla')
parser.add_argument('--audio', type=str, default='audio.wav', help='Äänitiedoston polku')
parser.add_argument('--rttm', type=str, default='audio1.rttm', help='RTTM-tiedoston polku')
parser.add_argument('--output', type=str, default='transcription.txt', help='Tulostiedoston polku')
parser.add_argument('--language', type=str, default='fi', help='Kielen koodi (esim. fi, en, sv)')
args = parser.parse_args()

# Aseta laitteen ja tarkkuuden asetukset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Käytetään laitetta: {device}")
print(f"Transkriptoidaan tiedostoa: {args.audio}")
print(f"Käytetään puhujien erottelua tiedostosta: {args.rttm}")

# Tarkista, että tiedostot ovat olemassa
if not os.path.exists(args.audio):
    raise FileNotFoundError(f"Äänitiedostoa ei löydy: {args.audio}")
if not os.path.exists(args.rttm):
    raise FileNotFoundError(f"RTTM-tiedostoa ei löydy: {args.rttm}")

# Lataa äänitiedosto
audio_input, sample_rate = sf.read(args.audio)

# Lataa puhujien erottelun tulokset
rttm = load_rttm(args.rttm)
diarization = next(iter(rttm.values()))

# Lataa prosessori ja malli (uudempi turbo-malli)
model_id = "openai/whisper-large-v3-turbo"

# Tarkista onko Flash Attention 2 saatavilla
try:
    import flash_attn
    has_flash_attn = True
    print("Flash Attention 2 on saatavilla, käytetään sitä.")
except ImportError:
    has_flash_attn = False
    print("Flash Attention 2 ei ole saatavilla, käytetään tavallista huomiomekanismia.")

# Tarkista onko torch.compile saatavilla (PyTorch 2.0+)
use_torch_compile = False
try:
    if torch.__version__ >= "2.0.0":
        use_torch_compile = True
        print(f"PyTorch {torch.__version__} tukee torch.compile-optimointia.")
except:
    print("PyTorch-versio ei tue torch.compile-optimointia.")

# Lataa malli optimoiduilla asetuksilla
if has_flash_attn:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2"
    )
else:
    # Tarkista onko SDPA saatavilla (PyTorch 2.1.1+)
    from transformers.utils import is_torch_sdpa_available
    if is_torch_sdpa_available():
        print("PyTorch SDPA on saatavilla, käytetään sitä.")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        )
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

# Käytä torch.compile-optimointia, jos saatavilla
if use_torch_compile and not has_flash_attn:  # torch.compile ei ole yhteensopiva Flash Attention 2:n kanssa
    try:
        print("Optimoidaan mallia torch.compile-toiminnolla...")
        # Aseta staattinen välimuisti ja käännä forward-funktio
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = 448
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        print("Malli optimoitu torch.compile-toiminnolla.")
    except Exception as e:
        print(f"torch.compile-optimointi epäonnistui: {e}")

# Siirrä malli GPU:lle, jos se on käytettävissä
model.to(device)

# Lataa prosessori
processor = AutoProcessor.from_pretrained(model_id)

# Aseta kieli ja tehtävä
language = args.language
task = "transcribe"
print(f"Käytetään kieltä: {language}")

transcription = ""

# Iteroi segmenttien yli
for segment, _, speaker in diarization.itertracks(yield_label=True):
    start = int(segment.start * sample_rate)
    end = int(segment.end * sample_rate)
    audio_chunk = audio_input[start:end]

    # Esikäsittele äänidata
    inputs = processor(
        audio_chunk,
        sampling_rate=sample_rate,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True
    )

    # Siirrä syötteet laitteelle
    inputs = inputs.to(device, dtype=torch_dtype)

    # Generoi transkriptio
    gen_kwargs = {
        "language": language,
        "task": task,
        "max_new_tokens": 448,
        "num_beams": 1
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Dekoodaa transkriptio
    chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Lisää transkriptio kokonaistranskriptioon puhujan kanssa
    transcription += f"{speaker}: {chunk_transcription}\n"

    print(f"Puhuja {speaker} segmentti {segment.start:.2f}-{segment.end:.2f} transkriboitu.")

# Tallenna transkriptio tekstitiedostoon
with open(args.output, 'w', encoding='utf-8') as f:
    f.write(transcription)

print(f"Transkriptio puhujien erottelulla valmis. Tulokset on tallennettu tiedostoon '{args.output}'.")
