import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import threading
import queue
import os
import time
import numpy as np
import soundfile as sf
from speaker_diarization import SpeakerDiarization

class Transcriber:
    def __init__(self, model_id="openai/whisper-small", language="fi", callback=None, use_diarization=True):
        """
        Initialize the transcriber.

        Args:
            model_id: The Whisper model ID to use
            language: The language code for transcription
            callback: Function to call when transcription is complete
            use_diarization: Whether to use speaker diarization
        """
        print(f"Alustetaan Transcriber, malli: {model_id}, kieli: {language}")
        self.model_id = model_id
        self.language = language
        self.callback = callback
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None
        self.transcription_queue = queue.Queue()
        self.processing = False
        self.thread = None
        self.use_diarization = use_diarization
        self.diarization = None
        self.model_loaded = False

        # Tarkista PyTorch-versio
        print(f"PyTorch-versio: {torch.__version__}")
        print(f"CUDA saatavilla: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA-versio: {torch.version.cuda}")
            print(f"CUDA-laitteet: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Laite {i}: {torch.cuda.get_device_name(i)}")

        # Load the model and processor
        self._load_model()

        # Initialize speaker diarization if enabled
        if self.use_diarization:
            self._init_diarization()

    def _init_diarization(self):
        """Initialize speaker diarization."""
        try:
            self.diarization = SpeakerDiarization()
        except Exception as e:
            print(f"Virhe puhujan tunnistuksen alustamisessa: {e}")
            self.use_diarization = False

    def _load_model(self):
        """Load the Whisper model and processor."""
        print(f"Ladataan Whisper-mallia: {self.model_id}")
        print(f"Käytetään laitetta: {self.device}")

        try:
            # Käytä pienempää mallia, joka latautuu nopeammin
            print("Käytetään pientä Whisper-mallia (whisper-small)")
            self.model_id = "openai/whisper-small"

            # Lataa malli yksinkertaisella tavalla
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            )

            print(f"Malli ladattu: {self.model}")  # Lisätty tulostus
            # Siirrä malli oikealle laitteelle
            self.model.to(self.device)

            # Lataa prosessori
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            print(f"Prosessori ladattu: {self.processor}")  # Lisätty tulostus

            # Merkitse malli ladatuksi
            self.model_loaded = True

            print("Malli ladattu onnistuneesti")

        except Exception as e:
            print(f"Virhe mallin lataamisessa: {e}")
            print("Transkriptio ei ole käytettävissä")

    def _process_audio_files(self):
        """Process audio files from the queue."""
        print("Transkriptioprosessi käynnistetty")
        while self.processing:
            try:
                # Get an audio file from the queue with a timeout
                audio_file = self.transcription_queue.get(timeout=0.1)
                print(f"Transkriptoidaan tiedostoa: {audio_file}")

                # Transcribe the audio file
                transcription = self.transcribe_file(audio_file)
                print(f"Transkriptio tulos: {transcription}")  # Lisätty tulostus

                # Call the callback function if provided
                if self.callback:
                    print(f"Kutsutaan takaisinkutsufunktiota transkriptiolle: {transcription[:50]}...")
                    self.callback(transcription, audio_file)
                else:
                    print("Takaisinkutsufunktiota ei ole määritetty")

                # Mark the task as done
                self.transcription_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Virhe äänitiedoston käsittelyssä: {e}")
                if self.transcription_queue.unfinished_tasks > 0:
                    self.transcription_queue.task_done()

        print("Transkriptioprosessi pysäytetty")

    def start_processing(self):
        """Start the transcription processing thread."""
        if self.processing:
            return

        self.processing = True

        # Start the processing thread
        self.thread = threading.Thread(target=self._process_audio_files)
        self.thread.daemon = True
        self.thread.start()

        print("Transcription processing started")

    def stop_processing(self):
        """Stop the transcription processing thread."""
        if not self.processing:
            return

        self.processing = False

        # Wait for the processing thread to finish
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        print("Transcription processing stopped")

    def add_audio_file(self, audio_file):
        """Add an audio file to the transcription queue."""
        if not os.path.exists(audio_file):
            print(f"Audio file does not exist: {audio_file}")
            return

        self.transcription_queue.put(audio_file)

    def transcribe_file(self, audio_file):
        """Transcribe an audio file."""
        try:
            # Load the audio file
            audio_input, sample_rate = sf.read(audio_file)

            # Check if we should use speaker diarization
            if self.use_diarization and self.diarization is not None:
                return self._transcribe_with_diarization(audio_input, sample_rate, audio_file)
            else:
                return self._transcribe_audio(audio_input, sample_rate)

        except Exception as e:
            print(f"Virhe tiedoston transkriptiossa: {e}")
            return f"Virhe: {e}"

    def _transcribe_with_diarization(self, audio_input, sample_rate, audio_file=None):
        """Transcribe audio with speaker diarization."""
        try:
            # Process speaker diarization
            if audio_file is not None and os.path.exists(audio_file):
                # Use the file directly if it exists
                speaker_turns = self.diarization.process_audio_file(audio_file)
            else:
                # Process the audio data
                speaker_turns = self.diarization.process_audio(audio_input, sample_rate)

            if not speaker_turns:
                print("Puhujan tunnistus ei onnistunut, käytetään tavallista transkriptiota.")
                return self._transcribe_audio(audio_input, sample_rate)

            # Transcribe each speaker segment
            full_transcription = ""

            for start, end, speaker in speaker_turns:
                # Extract the audio segment
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment = audio_input[start_sample:end_sample]

                # Skip segments that are too short
                if len(segment) < 0.5 * sample_rate:  # Skip segments shorter than 0.5 seconds
                    continue

                # Transcribe the segment
                segment_transcription = self._transcribe_audio(segment, sample_rate)

                # Add to the full transcription with speaker information
                if segment_transcription.strip():
                    full_transcription += f"{speaker}: {segment_transcription}\n\n"

            return full_transcription

        except Exception as e:
            print(f"Virhe puhujan tunnistuksessa: {e}")
            # Fall back to regular transcription
            return self._transcribe_audio(audio_input, sample_rate)

    def _transcribe_audio(self, audio_input, sample_rate):
        """Transcribe audio data."""
        # Tarkista, että malli on ladattu
        if not self.model_loaded or self.model is None or self.processor is None:
            print("Mallia ei ole ladattu, transkriptio ei ole mahdollista")
            return "Transkriptio ei ole käytettävissä. Mallia ei ole ladattu."

        try:
            # Varmista, että audio_input on oikean muotoinen
            if len(audio_input) == 0:
                print("Tyhjä äänisyöte")
                return "Tyhjä äänisyöte"

            # Tarkista äänen taso
            audio_level = np.max(np.abs(audio_input))
            print(f"Äänitaso transkriptiossa: {audio_level:.6f}")

            if audio_level < 0.001:  # Jos äänitaso on liian matala
                print("Äänitaso on liian matala transkriptiota varten")
                return "Äänitaso on liian matala. Puhu kovempaa tai tarkista mikrofoni."

            # Normalisoi ääni
            audio_input = audio_input / np.max(np.abs(audio_input)) if np.max(np.abs(audio_input)) > 0 else audio_input

            # Tulosta äänen tiedot
            print(f"Äänen muoto: {audio_input.shape}, näytteenottotaajuus: {sample_rate}")

            # Käsittele ääni
            try:
                inputs = self.processor(
                    audio_input,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    return_attention_mask=True
                )
                print("Ääni käsitelty prosessorilla onnistuneesti")
            except Exception as proc_error:
                print(f"Virhe äänen käsittelyssä prosessorilla: {proc_error}")
                return f"Virhe äänen käsittelyssä: {proc_error}"

            # Siirrä syötteet oikealle laitteelle
            try:
                inputs = inputs.to(self.device, dtype=self.torch_dtype)
                print(f"Syötteet siirretty laitteelle: {self.device}")
            except Exception as device_error:
                print(f"Virhe syötteiden siirrossa laitteelle: {device_error}")
                return f"Virhe syötteiden siirrossa: {device_error}"

            # Generoi transkriptio
            gen_kwargs = {
                "language": self.language,
                "task": "transcribe",
                "max_new_tokens": 256,  # Pienempi arvo nopeuttaa
                "num_beams": 1
            }

            try:
                with torch.no_grad():
                    print("Aloitetaan transkription generointi...")
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    print("Transkription generointi onnistui")
            except Exception as gen_error:
                print(f"Virhe transkription generoinnissa: {gen_error}")
                return f"Virhe transkription generoinnissa: {gen_error}"

            # Dekoodaa transkriptio
            try:
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f"Transkriptio dekoodattu: {transcription}")
            except Exception as decode_error:
                print(f"Virhe transkription dekoodauksessa: {decode_error}")
                return f"Virhe transkription dekoodauksessa: {decode_error}"

            # Tarkista, onko transkriptio tyhjä
            if not transcription.strip():
                print("Transkriptio on tyhjä")
                return "Ei tunnistettavaa puhetta. Puhu kovempaa tai tarkista mikrofoni."

            print(f"Transkriptio onnistui: {transcription}")
            return transcription

        except Exception as e:
            print(f"Virhe transkriptiossa: {e}")
            return f"Virhe: {e}"

    def set_language(self, language):
        """Set the language for transcription."""
        self.language = language
        print(f"Language set to: {language}")

    def get_queue_size(self):
        """Get the current size of the transcription queue."""
        return self.transcription_queue.qsize()
