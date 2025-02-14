"""
Model configuration for the Quran Reciter Identification project.
"""

# Audio Processing Configuration
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'win_length': 2048,
    'f_min': 20,
    'f_max': 8000,
    'power': 2.0
}

# Model Architecture Configuration
MODEL_CONFIG = {
    # Input dimensions
    'input_channels': 1,
    'num_classes': 20,  # Number of reciters to identify

    # CNN Architecture
    'conv_blocks': [
        # (channels, kernel_size, stride, padding)
        {'channels': 32,  'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'channels': 64,  'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    ],

    # Pooling
    'pool_size': 2,
    'pool_stride': 2,

    # Normalization and Regularization
    'batch_norm': True,
    'dropout_rate': 0.5,

    # Dense layers
    'dense_layers': [512, 256],

    # Activation functions
    'conv_activation': 'relu',
    'dense_activation': 'relu',
    'output_activation': 'softmax'
}

# Training Configuration
TRAINING_CONFIG = {
    # Optimization
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 100,

    # Learning rate scheduling
    'lr_scheduler': 'reduce_on_plateau',
    'lr_patience': 5,
    'lr_factor': 0.5,
    'lr_min': 1e-6,

    # Early stopping
    'early_stopping': True,
    'patience': 10,
    'min_delta': 0.001,

    # Loss function
    'loss': 'cross_entropy',

    # Mixed precision training
    'mixed_precision': True,

    # Gradient clipping
    'clip_grad_norm': 1.0
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'time_stretch': {
        'enabled': True,
        'min_rate': 0.9,
        'max_rate': 1.1,
        'probability': 0.5
    },
    'pitch_shift': {
        'enabled': True,
        'min_steps': -2,
        'max_steps': 2,
        'probability': 0.5
    },
    'noise': {
        'enabled': True,
        'min_noise_factor': 0.001,
        'max_noise_factor': 0.015,
        'probability': 0.3
    },
    'time_masking': {
        'enabled': True,
        'max_mask_size': 30,
        'num_masks': 2,
        'probability': 0.5
    },
    'freq_masking': {
        'enabled': True,
        'max_mask_size': 20,
        'num_masks': 2,
        'probability': 0.5
    }
}
