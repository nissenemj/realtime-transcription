import numpy as np
import soundfile as sf

class SpeakerDiarization:
    def __init__(self):
        """
        Initialize the simple speaker diarization module.

        This is a simplified version that doesn't require external models.
        It uses basic audio processing to segment audio by silence.
        """
        self.min_silence_duration = 0.5  # Minimum silence duration in seconds
        self.energy_threshold = 0.05     # Energy threshold for silence detection
        self.min_segment_duration = 1.0  # Minimum segment duration in seconds

        print("Yksinkertainen puhujan tunnistus alustettu.")

    def _detect_segments(self, audio_data, sample_rate):
        """
        Detect segments in audio based on silence.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            List of (start, end) tuples
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Calculate energy
        energy = np.abs(audio_data)

        # Find silence regions
        is_silence = energy < self.energy_threshold

        # Convert to samples
        min_silence_samples = int(self.min_silence_duration * sample_rate)
        min_segment_samples = int(self.min_segment_duration * sample_rate)

        # Find silence regions that are long enough
        silence_starts = []
        silence_ends = []
        in_silence = False
        silence_start = 0

        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                # Start of silence
                silence_start = i
                in_silence = True
            elif not silent and in_silence:
                # End of silence
                if i - silence_start >= min_silence_samples:
                    silence_starts.append(silence_start)
                    silence_ends.append(i)
                in_silence = False

        # Add the last silence if needed
        if in_silence and len(audio_data) - silence_start >= min_silence_samples:
            silence_starts.append(silence_start)
            silence_ends.append(len(audio_data))

        # Create segments from silence boundaries
        segments = []

        # Add the first segment if needed
        if not silence_starts or silence_starts[0] > 0:
            start = 0
            end = silence_starts[0] if silence_starts else len(audio_data)
            if end - start >= min_segment_samples:
                segments.append((start / sample_rate, end / sample_rate))

        # Add segments between silences
        for i in range(len(silence_ends)):
            start = silence_ends[i]
            end = silence_starts[i+1] if i+1 < len(silence_starts) else len(audio_data)
            if end - start >= min_segment_samples:
                segments.append((start / sample_rate, end / sample_rate))

        return segments

    def process_audio(self, audio_data, sample_rate):
        """
        Process audio data for speaker diarization.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            List of (start, end, speaker) tuples
        """
        try:
            # Detect segments based on silence
            segments = self._detect_segments(audio_data, sample_rate)

            # Assign speaker labels
            speaker_turns = []
            for i, (start, end) in enumerate(segments):
                # Alternate between SPEAKER_00 and SPEAKER_01
                speaker = f"SPEAKER_{i % 2:02d}"
                speaker_turns.append((start, end, speaker))

            return speaker_turns

        except Exception as e:
            print(f"Virhe puhujan tunnistuksessa: {e}")
            return []

    def process_audio_file(self, audio_file):
        """
        Process an audio file for speaker diarization.

        Args:
            audio_file: Path to the audio file

        Returns:
            List of (start, end, speaker) tuples
        """
        try:
            # Load the audio file
            audio_data, sample_rate = sf.read(audio_file)

            # Process the audio data
            return self.process_audio(audio_data, sample_rate)

        except Exception as e:
            print(f"Virhe puhujan tunnistuksessa: {e}")
            return []

    def get_speaker_segments(self, audio_data, sample_rate, min_segment_duration=1.0):
        """
        Get audio segments by speaker.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            min_segment_duration: Minimum duration of a segment in seconds

        Returns:
            List of (audio_segment, start, end, speaker) tuples
        """
        speaker_turns = self.process_audio(audio_data, sample_rate)

        segments = []
        for start, end, speaker in speaker_turns:
            # Skip segments that are too short
            if end - start < min_segment_duration:
                continue

            # Convert time to samples
            start_sample = int(start * sample_rate)
            end_sample = min(int(end * sample_rate), len(audio_data))  # Ensure we don't go out of bounds

            # Extract the audio segment
            segment = audio_data[start_sample:end_sample]

            segments.append((segment, start, end, speaker))

        return segments
