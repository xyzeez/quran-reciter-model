"""
Pipeline configuration for the Quran Reciter Identification project.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATASETS_DIR = BASE_DIR / 'datasets'
LOGS_DIR = BASE_DIR / 'logs'
RUNS_DIR = BASE_DIR / 'runs'
CACHE_DIR = BASE_DIR / 'cache'

# Dataset paths
RAW_DATA_DIR = DATASETS_DIR / 'raw'
PROCESSED_DATA_DIR = DATASETS_DIR / 'processed'

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    # Audio loading
    'min_duration': 5,    # Minimum audio duration in seconds
    'max_duration': 30,   # Maximum audio duration in seconds
    'skip_start': 5,      # Skip first N seconds (bismillah)

    # Audio processing
    'remove_silence': True,
    'silence_threshold': -60,  # dB
    'min_silence_duration': 0.3,  # seconds

    # Feature extraction
    'normalize_audio': True,
    'normalize_features': True,
    'feature_type': 'mel_spectrogram',

    # Segment generation
    'segment_duration': 10,  # seconds
    'segment_overlap': 2,    # seconds
    'min_segments': 3,       # Minimum segments per file

    # Parallel processing
    'num_workers': 4,
    'batch_size': 32
}

# Dataset Configuration
DATASET_CONFIG = {
    # Data split ratios
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    'shuffle': True,
    'drop_last': True,

    # Caching
    'use_cache': True,
    'cache_size': 1000,  # Number of items to keep in memory

    # Balancing
    'balance_classes': True,
    'min_samples_per_class': 100,
    'max_samples_per_class': 1000
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    # Checkpointing
    'save_best_only': True,
    'save_frequency': 5,  # Save every N epochs
    'max_checkpoints': 3,  # Maximum number of checkpoints to keep

    # Validation
    'validate_frequency': 1,  # Validate every N epochs
    'eval_metrics': ['accuracy', 'precision', 'recall', 'f1'],

    # TensorBoard logging
    'log_frequency': 50,  # Log every N batches
    'log_images': True,   # Log spectrograms
    'log_gradients': True,

    # Device settings
    'device': 'cuda',  # 'cuda' or 'cpu'
    'gpu_ids': [0],    # List of GPU ids to use

    # Random seed
    'seed': 42
}

# Create required directories
for directory in [LOGS_DIR, RUNS_DIR, CACHE_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
