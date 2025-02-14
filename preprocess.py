"""
Preprocessing script for the Quran Reciter Identification project.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any

from config.pipeline_config import PREPROCESSING_CONFIG
from utils.logging.logger import PipelineLogger
from utils import file_manager
from data.data_loader import AudioLoader


def preprocess_reciter(
    reciter_dir: Path,
    output_dir: Path,
    audio_loader: AudioLoader,
    logger: PipelineLogger
) -> Dict[str, Any]:
    """Process all audio files for a single reciter.

    Args:
        reciter_dir: Path to reciter's directory
        output_dir: Path to output directory
        audio_loader: AudioLoader instance
        logger: Logger instance

    Returns:
        Statistics dictionary for the reciter
    """
    # Get all audio files
    audio_files = list(reciter_dir.glob("*.mp3"))
    process_task = logger.create_task(
        f"Processing {reciter_dir.name}",
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
        audio, sr, error = audio_loader.load_audio(audio_file)
        if error:
            logger.log_warning(error)
            stats['failed_files'].append({
                'file': str(audio_file),
                'error': error
            })
            logger.update_task(process_task)
            continue

        # Process audio
        audio = audio_loader.process_audio(audio, sr)

        # Extract features
        features = audio_loader.extract_features(audio, sr)

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

    return stats


def main():
    """Main preprocessing pipeline."""
    with PipelineLogger("preprocess") as logger:
        try:
            # Initialize components
            audio_loader = AudioLoader(PREPROCESSING_CONFIG)

            # Get dataset paths
            raw_dir = file_manager.get_raw_dataset_path()
            logger.log_info(f"Using dataset from: {raw_dir}")

            # Clean processed directory
            logger.log_info("Cleaning processed directory...")
            file_manager.clean_processed_dir()
            processed_dir = file_manager.get_processed_dir()

            # Get list of reciters
            reciters = file_manager.get_reciters(raw_dir)
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
                output_dir = file_manager.create_reciter_output_dir(
                    reciter_name)

                # Process reciter
                stats = preprocess_reciter(
                    reciter_dir, output_dir, audio_loader, logger)

                # Update metadata
                metadata['reciters'][reciter_name] = stats

                # Update overall progress
                logger.update_task(overall_task)

                # Log system info
                logger.log_system_info()

            # Save metadata
            file_manager.save_metadata(metadata, processed_dir)
            logger.log_success("Preprocessing completed successfully")

        except Exception as e:
            logger.log_error(f"Preprocessing failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
