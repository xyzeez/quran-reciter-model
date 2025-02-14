"""
Preprocessing script for the Quran Reciter Identification project.
"""

import librosa
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Optional
import torch
from tqdm import tqdm

from config.model_config import AUDIO_CONFIG
from config.pipeline_config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREPROCESSING_CONFIG
)
from utils.logging.logger import PipelineLogger


def load_audio(
    file_path: Path,
    config: dict = PREPROCESSING_CONFIG
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Load and validate audio file.

    Args:
        file_path: Path to audio file
        config: Preprocessing configuration

    Returns:
        Tuple of (audio_data, sample_rate, error_message)
    """
    try:
        # Load audio with offset to skip bismillah
        y, sr = librosa.load(
            file_path,
            sr=AUDIO_CONFIG['sample_rate'],
            offset=config['skip_start'],
            duration=config['max_duration']
        )

        # Check duration
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < config['min_duration']:
            return None, None, f"Audio too short: {duration:.2f}s"

        return y, sr, None

    except Exception as e:
        return None, None, f"Error loading audio: {str(e)}"


def process_audio(
    audio: np.ndarray,
    sr: int,
    config: dict = PREPROCESSING_CONFIG
) -> np.ndarray:
    """Process audio data.

    Args:
        audio: Audio time series
        sr: Sample rate
        config: Preprocessing configuration

    Returns:
        Processed audio data
    """
    # Remove silence
    if config['remove_silence']:
        non_silent = librosa.effects.split(
            audio,
            top_db=abs(config['silence_threshold']),
            frame_length=2048,
            hop_length=512
        )
        audio = np.concatenate([audio[start:end] for start, end in non_silent])

    # Normalize audio
    if config['normalize_audio']:
        audio = librosa.util.normalize(audio)

    return audio


def extract_features(
    audio: np.ndarray,
    sr: int,
    config: dict = AUDIO_CONFIG
) -> np.ndarray:
    """Extract mel spectrogram features.

    Args:
        audio: Audio time series
        sr: Sample rate
        config: Audio configuration

    Returns:
        Mel spectrogram features
    """
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=config['n_mels'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        win_length=config['win_length'],
        window='hann',
        center=True,
        pad_mode='reflect',
        power=config['power'],
        fmin=config['f_min'],
        fmax=config['f_max']
    )

    # Convert to log scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize features
    if PREPROCESSING_CONFIG['normalize_features']:
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    return mel_spec


def main():
    """Main preprocessing pipeline."""
    with PipelineLogger("preprocess") as logger:
        try:
            # Start pipeline
            progress = logger.start_pipeline()

            # Get list of reciters
            reciters = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
            total_reciters = len(reciters)

            logger.log_info(f"Found {total_reciters} reciters")
            overall_task = logger.create_task(
                "Processing Reciters",
                total=total_reciters
            )

            # Process each reciter
            metadata = {'reciters': {}}

            for reciter_dir in reciters:
                reciter_name = reciter_dir.name
                logger.log_info(f"Processing reciter: {reciter_name}")

                # Create output directory
                output_dir = PROCESSED_DATA_DIR / reciter_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Get all audio files
                audio_files = list(reciter_dir.glob("*.mp3"))
                process_task = logger.create_task(
                    f"Processing {reciter_name}",
                    total=len(audio_files)
                )

                # Initialize statistics
                stats = {
                    'total_files': len(audio_files),
                    'processed_files': 0,
                    'failed_files': [],
                    'total_duration': 0.0
                }

                # Process audio files
                for audio_file in audio_files:
                    # Load audio
                    audio, sr, error = load_audio(audio_file)
                    if error:
                        logger.log_warning(error)
                        stats['failed_files'].append({
                            'file': str(audio_file),
                            'error': error
                        })
                        logger.update_task(process_task)
                        continue

                    # Process audio
                    audio = process_audio(audio, sr)

                    # Extract features
                    features = extract_features(audio, sr)

                    # Save features
                    output_file = output_dir / f"{audio_file.stem}.npy"
                    np.save(output_file, features)

                    # Update statistics
                    stats['processed_files'] += 1
                    stats['total_duration'] += len(audio) / sr

                    # Update progress
                    logger.update_task(process_task)

                # Log statistics
                logger.log_stats({
                    'Total Files': stats['total_files'],
                    'Processed Files': stats['processed_files'],
                    'Failed Files': len(stats['failed_files']),
                    'Total Duration': f"{stats['total_duration']:.2f}s"
                })

                # Update metadata
                metadata['reciters'][reciter_name] = stats

                # Update overall progress
                logger.update_task(overall_task)

                # Log system info
                logger.log_system_info()

            # Save metadata
            with open(PROCESSED_DATA_DIR / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.log_success("Preprocessing completed successfully")

        except Exception as e:
            logger.log_error(f"Preprocessing failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
