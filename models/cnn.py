"""
CNN model architecture for Quran reciter identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm  # No bias when using batch norm
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(
            out_channels) if use_batch_norm else None

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ReciterCNN(nn.Module):
    """CNN model for Quran reciter identification."""

    def __init__(self, config: dict):
        super().__init__()

        # Extract configuration
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']
        self.conv_blocks = config['conv_blocks']
        self.pool_size = config['pool_size']
        self.pool_stride = config['pool_stride']
        self.dense_layers = config['dense_layers']
        self.dropout_rate = config['dropout_rate']
        self.batch_norm = config['batch_norm']
        self.conv_activation = config['conv_activation']

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels

        for block in self.conv_blocks:
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=block['channels'],
                    kernel_size=block['kernel_size'],
                    stride=block['stride'],
                    padding=block['padding'],
                    use_batch_norm=self.batch_norm,
                    activation=self.conv_activation
                )
            )
            in_channels = block['channels']

        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=self.pool_size,
            stride=self.pool_stride
        )

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate dense input size
        self._dense_input_size = in_channels * 4 * 4

        # Build dense layers
        self.dense_layers = nn.ModuleList()
        dense_sizes = [self._dense_input_size] + self.dense_layers

        for i in range(len(dense_sizes) - 1):
            self.dense_layers.extend([
                nn.Linear(dense_sizes[i], dense_sizes[i + 1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(dense_sizes[i + 1]),
                nn.Dropout(self.dropout_rate)
            ])

        # Output layer
        self.output = nn.Linear(dense_sizes[-1], self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, freq, time)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        for conv in self.conv_layers:
            x = conv(x)
            x = self.pool(x)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)

        # Output layer
        x = self.output(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction probabilities and classes.

        Args:
            x: Input tensor

        Returns:
            Tuple of (probabilities, predicted_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.exp(logits)
            _, predicted = torch.max(probs, 1)
            return probs, predicted
