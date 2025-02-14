"""
Training script for the Quran Reciter Identification project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
from datetime import datetime
import json

from models.cnn import ReciterCNN
from data.dataset import create_dataloaders
from config.model_config import MODEL_CONFIG, TRAINING_CONFIG
from config.pipeline_config import EXPERIMENT_CONFIG
from utils.logging.logger import PipelineLogger
from utils import file_manager


class Trainer:
    """Model trainer class."""

    def __init__(self, model: nn.Module, config: dict = TRAINING_CONFIG):
        """Initialize trainer."""
        self.model = model
        self.config = config

        # Setup device
        self.device = torch.device(EXPERIMENT_CONFIG['device'])
        if self.device.type == 'cuda':
            self.model = nn.DataParallel(
                self.model,
                device_ids=EXPERIMENT_CONFIG['gpu_ids']
            )
        self.model.to(self.device)

        # Setup optimizer
        if config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lr_factor'],
            patience=config['lr_patience'],
            min_lr=config['lr_min']
        )

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()

        # Setup mixed precision training
        self.scaler = GradScaler() if config['mixed_precision'] else None

        # Setup metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create run directory
        self.run_dir = file_manager.create_run_directory()

        # Save configurations
        self.save_configs()

    def save_configs(self):
        """Save experiment configurations."""
        configs = {
            'model_config': MODEL_CONFIG,
            'training_config': TRAINING_CONFIG,
            'experiment_config': EXPERIMENT_CONFIG
        }

        for name, config in configs.items():
            file_manager.save_metadata(config, str(
                self.run_dir / 'configs'), f"{name}.json")

    def train_epoch(self, train_loader, logger) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            logger: Pipeline logger

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Create progress tracking
        train_task = logger.create_task(
            "Training Batch",
            total=len(train_loader)
        )

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.config['mixed_precision']:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['clip_grad_norm']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['clip_grad_norm']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Standard backward pass
                loss.backward()

                # Gradient clipping
                if self.config['clip_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['clip_grad_norm']
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

            # Log batch metrics
            if batch_idx % EXPERIMENT_CONFIG['log_frequency'] == 0:
                logger.log_stats({
                    'Batch Loss': f"{loss.item():.4f}",
                    'Batch Accuracy': f"{100. * total_correct / total_samples:.2f}%",
                    'Learning Rate': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

            # Update progress
            logger.update_task(train_task)

        return total_loss / len(train_loader)

    def validate(self, val_loader, logger) -> float:
        """Validate the model.

        Args:
            val_loader: Validation data loader
            logger: Pipeline logger

        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # Create progress tracking
        val_task = logger.create_task(
            "Validation",
            total=len(val_loader)
        )

        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)

                # Update progress
                logger.update_task(val_task)

        val_loss /= len(val_loader)
        accuracy = 100. * correct / total

        logger.log_stats({
            'Validation Loss': f"{val_loss:.4f}",
            'Validation Accuracy': f"{accuracy:.2f}%"
        })

        return val_loss

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
        """
        checkpoint_dir = self.run_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss
        }

        # Save latest checkpoint
        torch.save(
            checkpoint,
            checkpoint_dir / 'last_checkpoint.pt'
        )

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(
                checkpoint,
                checkpoint_dir / 'best_checkpoint.pt'
            )
            return True
        return False

    def train(self, train_loader, val_loader, logger):
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            logger: Pipeline logger
        """
        logger.log_info("Starting training...")

        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            # Training phase
            logger.log_info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            train_loss = self.train_epoch(train_loader, logger)

            # Validation phase
            val_loss = self.validate(val_loader, logger)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save checkpoint
            is_best = self.save_checkpoint(epoch, val_loss)

            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.log_stats({
                'Epoch': epoch + 1,
                'Training Loss': f"{train_loss:.4f}",
                'Validation Loss': f"{val_loss:.4f}",
                'Epoch Time': f"{epoch_time:.2f}s",
                'Best Model': "Yes" if is_best else "No"
            })

            # Early stopping
            if is_best:
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (self.config['early_stopping'] and
                    self.patience_counter >= self.config['patience']):
                logger.log_info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                break

            # Log system info
            logger.log_system_info()


def main():
    """Main training pipeline."""
    with PipelineLogger("train") as logger:
        try:
            # Create model
            logger.log_info("Creating model...")
            model = ReciterCNN(MODEL_CONFIG)

            # Create data loaders
            logger.log_info("Creating data loaders...")
            processed_dir = file_manager._get_dir_path('datasets')['processed']
            data_loaders = create_dataloaders(
                processed_dir,
                batch_size=TRAINING_CONFIG['batch_size']
            )

            # Create trainer
            trainer = Trainer(model)

            # Train model
            trainer.train(
                data_loaders['train'],
                data_loaders['val'],
                logger
            )

            # Clean up old runs
            file_manager.cleanup_old_runs(max_runs=5)

            logger.log_success("Training completed successfully")

        except Exception as e:
            logger.log_error(f"Training failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
