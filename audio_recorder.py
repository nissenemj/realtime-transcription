import sounddevice as sd
import numpy as np
import threading
import queue
import tempfile
import os
import wave
import time
from scipy.io import wavfile

class AudioRecorder:
    def __init__(self, callback=None, chunk_duration=3):
        """
        Initialize the audio recorder.

        Args:
            callback: Function to call when a chunk of audio is recorded
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.callback = callback
        self.chunk_duration = chunk_duration
        self.recording = False
        self.audio_queue = queue.Queue()
        self.devices = self._get_devices()
        self.sample_rate = 16000  # Sample rate for Whisper
        self.channels = 1  # Mono audio
        self.dtype = 'float32'  # Data type for audio
        self.temp_dir = tempfile.mkdtemp()
        self.current_device = None
        self.stream = None
        self.thread = None
        self.chunk_count = 0

    def _get_devices(self):
        """Get available audio devices."""
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device['name']))

        # Add special option for system audio on macOS
        if os.name == 'posix' and os.uname().sysname == 'Darwin':
            input_devices.append((-1, "Järjestelmän ääni (macOS)"))

        return input_devices

    def _setup_system_audio_macos(self):
        """Set up system audio capture on macOS using BlackHole."""
        try:
            # Check if BlackHole is installed
            devices = sd.query_devices()
            blackhole_exists = any("BlackHole" in device['name'] for device in devices)

            if not blackhole_exists:
                print("BlackHole-ajuria ei löydy. Järjestelmän äänen kaappaus ei välttämättä toimi.")
                print("Asenna BlackHole: https://existential.audio/blackhole/")
                return False

            # Find BlackHole device
            blackhole_id = None
            for i, device in enumerate(devices):
                if "BlackHole" in device['name']:
                    blackhole_id = i
                    break

            if blackhole_id is not None:
                # Set BlackHole as the current device
                self.current_device = blackhole_id
                print(f"Käytetään BlackHole-laitetta järjestelmän äänen kaappaukseen: {blackhole_id}")
                return True

            return False
        except Exception as e:
            print(f"Virhe järjestelmän äänen asetuksissa: {e}")
            return False

    def get_available_devices(self):
        """Return a list of available input devices."""
        return self.devices

    def _audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream."""
        if status:
            print(f"Status: {status}")

        # Add the audio data to the queue
        self.audio_queue.put(indata.copy())

    def _process_audio(self):
        """Process audio chunks from the queue."""
        chunk_data = []
        chunk_samples = 0
        target_samples = int(self.chunk_duration * self.sample_rate)

        print(f"Äänen käsittely aloitettu. Tavoite näytteenottotaajuus: {self.sample_rate} Hz, tavoite näytteiden määrä: {target_samples}")

        while self.recording:
            try:
                # Get audio data from the queue with a timeout
                data = self.audio_queue.get(timeout=0.1)
                chunk_data.append(data)
                chunk_samples += len(data)

                # Print debug info occasionally
                if self.chunk_count % 10 == 0:
                    print(f"Ääntä vastaanotettu: {chunk_samples}/{target_samples} näytettä")

                # If we have enough samples for a chunk, process it
                if chunk_samples >= target_samples:
                    # Concatenate all the data
                    audio_chunk = np.concatenate(chunk_data)

                    # Save the chunk to a temporary WAV file
                    chunk_filename = os.path.join(self.temp_dir, f"chunk_{self.chunk_count}.wav")
                    self._save_wav(chunk_filename, audio_chunk)

                    print(f"Äänipalanen {self.chunk_count} tallennettu tiedostoon: {chunk_filename}")

                    # Call the callback function if provided
                    if self.callback:
                        print(f"Kutsutaan takaisinkutsufunktiota äänipalaselle {self.chunk_count}")
                        self.callback(chunk_filename)
                    else:
                        print("Takaisinkutsufunktiota ei ole määritetty")

                    # Reset for the next chunk
                    chunk_data = []
                    chunk_samples = 0
                    self.chunk_count += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Virhe äänen käsittelyssä: {e}")
                break

        print("Äänen käsittely lopetettu.")

    def _save_wav(self, filename, audio_data):
        """Save audio data to a WAV file."""
        # Convert from float32 [-1.0, 1.0] to int16 [-32768, 32767]
        audio_data_int = (audio_data * 32767).astype(np.int16)

        # Save as WAV file
        wavfile.write(filename, self.sample_rate, audio_data_int)

    def start_recording(self, device_id=None):
        """Start recording audio."""
        if self.recording:
            return

        # Handle special case for system audio on macOS
        if device_id == -1:
            # Set up system audio capture on macOS
            if not self._setup_system_audio_macos():
                print("Järjestelmän äänen kaappaus epäonnistui.")
                return
        else:
            # Set the device
            if device_id is not None:
                self.current_device = device_id
            else:
                # Use the default input device
                self.current_device = sd.default.device[0]

        # Start the audio stream
        self.stream = sd.InputStream(
            device=self.current_device,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            dtype=self.dtype
        )

        self.recording = True
        self.stream.start()

        # Start the processing thread
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()

        print(f"Nauhoitus aloitettu laitteella: {self.current_device}")

    def stop_recording(self):
        """Stop recording audio."""
        if not self.recording:
            return

        self.recording = False

        # Stop the audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Wait for the processing thread to finish
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        print("Recording stopped")

        # Return the path to the temporary directory with all chunks
        return self.temp_dir

    def cleanup(self):
        """Clean up temporary files."""
        # Stop recording if still active
        if self.recording:
            self.stop_recording()

        # Remove temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))

        # Remove temporary directory
        os.rmdir(self.temp_dir)

        print("Cleanup completed")
