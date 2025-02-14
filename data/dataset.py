"""
Dataset module for the Quran Reciter Identification project.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

from config.model_config import AUDIO_CONFIG, AUGMENTATION_CONFIG
from config.pipeline_config import DATASET_CONFIG


class AudioAugmenter:
    """Audio augmentation class."""

    def __init__(self, config: dict = AUGMENTATION_CONFIG):
        self.config = config

    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Apply time stretching augmentation."""
        if (self.config['time_stretch']['enabled'] and
                random.random() < self.config['time_stretch']['probability']):
            rate = random.uniform(
                self.config['time_stretch']['min_rate'],
                self.config['time_stretch']['max_rate']
            )
            return librosa.effects.time_stretch(audio, rate=rate)
        return audio

    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply pitch shifting augmentation."""
        if (self.config['pitch_shift']['enabled'] and
                random.random() < self.config['pitch_shift']['probability']):
            steps = random.uniform(
                self.config['pitch_shift']['min_steps'],
                self.config['pitch_shift']['max_steps']
            )
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        return audio

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random noise to audio."""
        if (self.config['noise']['enabled'] and
                random.random() < self.config['noise']['probability']):
            noise_factor = random.uniform(
                self.config['noise']['min_noise_factor'],
                self.config['noise']['max_noise_factor']
            )
            noise = np.random.randn(len(audio))
            return audio + noise_factor * noise
        return audio

    def mask_time(self, spec: np.ndarray) -> np.ndarray:
        """Apply time masking to spectrogram."""
        if (self.config['time_masking']['enabled'] and
                random.random() < self.config['time_masking']['probability']):
            for _ in range(self.config['time_masking']['num_masks']):
                t = random.randint(
                    0, self.config['time_masking']['max_mask_size'])
                t0 = random.randint(0, spec.shape[1] - t)
                spec[:, t0:t0 + t] = spec.mean()
        return spec

    def mask_freq(self, spec: np.ndarray) -> np.ndarray:
        """Apply frequency masking to spectrogram."""
        if (self.config['freq_masking']['enabled'] and
                random.random() < self.config['freq_masking']['probability']):
            for _ in range(self.config['freq_masking']['num_masks']):
                f = random.randint(
                    0, self.config['freq_masking']['max_mask_size'])
                f0 = random.randint(0, spec.shape[0] - f)
                spec[f0:f0 + f] = spec.mean()
        return spec


class ReciterDataset(Dataset):
    """Dataset class for Quran reciter identification."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        mode: str = 'train',
        transform: Optional[AudioAugmenter] = None
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to processed data directory
            mode: One of 'train', 'val', or 'test'
            transform: Optional audio augmentation transforms
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Get reciter mapping
        self.reciters = sorted(list(self.metadata['reciters'].keys()))
        self.reciter_to_idx = {name: idx for idx,
                               name in enumerate(self.reciters)}

        # Load file paths and labels
        self.samples = []
        for reciter in self.reciters:
            reciter_dir = self.data_dir / reciter
            for file_path in reciter_dir.glob('*.npy'):
                self.samples.append({
                    'path': file_path,
                    'label': self.reciter_to_idx[reciter]
                })

        # Split dataset
        if mode != 'train':
            random.Random(42).shuffle(self.samples)
            n_total = len(self.samples)
            if mode == 'val':
                start_idx = int(n_total * DATASET_CONFIG['train_ratio'])
                end_idx = start_idx + \
                    int(n_total * DATASET_CONFIG['val_ratio'])
                self.samples = self.samples[start_idx:end_idx]
            else:  # test
                start_idx = int(n_total * (1 - DATASET_CONFIG['test_ratio']))
                self.samples = self.samples[start_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (spectrogram, label)
        """
        sample = self.samples[idx]

        # Load spectrogram
        spec = np.load(sample['path'])

        # Apply augmentation if in training mode
        if self.mode == 'train' and self.transform:
            spec = self.transform.mask_time(spec)
            spec = self.transform.mask_freq(spec)

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()

        # Add channel dimension
        spec_tensor = spec_tensor.unsqueeze(0)

        return spec_tensor, sample['label']


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = DATASET_CONFIG['batch_size']
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation and testing.

    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for data loading

    Returns:
        Dictionary containing data loaders for each split
    """
    # Create augmenter for training
    augmenter = AudioAugmenter()

    # Create datasets
    train_dataset = ReciterDataset(data_dir, mode='train', transform=augmenter)
    val_dataset = ReciterDataset(data_dir, mode='val')
    test_dataset = ReciterDataset(data_dir, mode='test')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATASET_CONFIG['num_workers'],
        pin_memory=DATASET_CONFIG['pin_memory'],
        drop_last=DATASET_CONFIG['drop_last']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATASET_CONFIG['num_workers'],
        pin_memory=DATASET_CONFIG['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATASET_CONFIG['num_workers'],
        pin_memory=DATASET_CONFIG['pin_memory']
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
