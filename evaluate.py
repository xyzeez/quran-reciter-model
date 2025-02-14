"""
Evaluation script for the Quran Reciter Identification project.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from models.cnn import ReciterCNN
from data.dataset import create_dataloaders
from config.model_config import MODEL_CONFIG
from utils.logging.logger import PipelineLogger
from utils import file_manager


def evaluate_model(model: nn.Module, test_loader, device: torch.device) -> dict:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: Path) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main evaluation pipeline."""
    with PipelineLogger("evaluate") as logger:
        try:
            # Load latest checkpoint
            logger.log_info("Loading latest checkpoint...")
            checkpoint_path = file_manager.get_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoint found")

            checkpoint = torch.load(checkpoint_path)

            # Create model and load weights
            model = ReciterCNN(MODEL_CONFIG)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Setup device
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()

            # Create data loader
            logger.log_info("Creating test data loader...")
            processed_dir = file_manager._get_dir_path('datasets')['processed']
            data_loaders = create_dataloaders(
                processed_dir,
                batch_size=32,
                test_only=True
            )

            # Evaluate model
            logger.log_info("Evaluating model...")
            results = evaluate_model(model, data_loaders['test'], device)

            # Get class names
            class_names = data_loaders['test'].dataset.classes

            # Calculate metrics
            report = classification_report(
                results['labels'],
                results['predictions'],
                target_names=class_names,
                output_dict=True
            )

            cm = confusion_matrix(results['labels'], results['predictions'])

            # Create metrics directory
            run_dir = checkpoint_path.parent.parent
            metrics_dir = run_dir / 'metrics'
            metrics_dir.mkdir(exist_ok=True)

            # Save metrics
            file_manager.save_metadata(report, str(
                metrics_dir), 'classification_report.json')
            plot_confusion_matrix(
                cm, class_names, metrics_dir / 'confusion_matrix.png')

            # Log results
            logger.log_stats({
                'Accuracy': f"{report['accuracy']:.4f}",
                'Macro Avg F1': f"{report['macro avg']['f1-score']:.4f}",
                'Weighted Avg F1': f"{report['weighted avg']['f1-score']:.4f}"
            })

            logger.log_success("Evaluation completed successfully")

        except Exception as e:
            logger.log_error(f"Evaluation failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
