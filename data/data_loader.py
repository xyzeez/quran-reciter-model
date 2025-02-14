"""
Data loading and feature extraction utilities.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from config.model_config import AUDIO_CONFIG
from config.pipeline_config import PREPROCESSING_CONFIG


class AudioLoader:
    """Handles loading and processing of audio files."""

    def __init__(self, config: Dict[str, Any] = PREPROCESSING_CONFIG):
        self.config = config
        self.audio_config = AUDIO_CONFIG

    def load_audio(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
        """Load and validate audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate, error_message)
        """
        try:
            # Load audio with offset to skip bismillah
            y, sr = librosa.load(
                file_path,
                sr=self.audio_config['sample_rate'],
                offset=self.config['skip_start'],
                duration=self.config['max_duration']
            )

            # Check duration
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < self.config['min_duration']:
                return None, None, f"Audio too short: {duration:.2f}s"

            return y, sr, None

        except Exception as e:
            return None, None, f"Error loading audio: {str(e)}"

    def process_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio data.

        Args:
            audio: Audio time series
            sr: Sample rate

        Returns:
            Processed audio data
        """
        # Remove silence
        if self.config['remove_silence']:
            non_silent = librosa.effects.split(
                audio,
                top_db=abs(self.config['silence_threshold']),
                frame_length=2048,
                hop_length=512
            )
            audio = np.concatenate([audio[start:end]
                                    for start, end in non_silent])

        # Normalize audio
        if self.config['normalize_audio']:
            audio = librosa.util.normalize(audio)

        return audio

    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel spectrogram features.

        Args:
            audio: Audio time series
            sr: Sample rate

        Returns:
            Mel spectrogram features
        """
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.audio_config['n_mels'],
            n_fft=self.audio_config['n_fft'],
            hop_length=self.audio_config['hop_length'],
            win_length=self.audio_config['win_length'],
            window='hann',
            center=True,
            pad_mode='reflect',
            power=self.audio_config['power'],
            fmin=self.audio_config['f_min'],
            fmax=self.audio_config['f_max']
        )

        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize features
        if self.config['normalize_features']:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec
