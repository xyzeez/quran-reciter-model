"""
Evaluation script for the Quran Reciter Identification project.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from models.cnn import ReciterCNN
from data.dataset import create_dataloaders
from config.model_config import MODEL_CONFIG
from config.pipeline_config import PROCESSED_DATA_DIR, RUNS_DIR, EXPERIMENT_CONFIG
from utils.logging.logger import PipelineLogger


class ModelEvaluator:
    """Model evaluation class."""

    def __init__(self, model: nn.Module, experiment_dir: Path):
        """Initialize evaluator.

        Args:
            model: Model to evaluate
            experiment_dir: Path to experiment directory
        """
        self.model = model
        self.exp_dir = experiment_dir

        # Setup device
        self.device = torch.device(EXPERIMENT_CONFIG['device'])
        if self.device.type == 'cuda':
            self.model = nn.DataParallel(
                self.model,
                device_ids=EXPERIMENT_CONFIG['gpu_ids']
            )
        self.model.to(self.device)

        # Create evaluation directory
        self.eval_dir = self.exp_dir / 'evaluation'
        self.eval_dir.mkdir(exist_ok=True)

        # Load reciter mapping
        with open(PROCESSED_DATA_DIR / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        self.reciters = sorted(list(metadata['reciters'].keys()))

    def evaluate(self, test_loader, logger) -> dict:
        """Evaluate the model.

        Args:
            test_loader: Test data loader
            logger: Pipeline logger

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        # Create progress tracking
        test_task = logger.create_task(
            "Evaluating",
            total=len(test_loader)
        )

        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Get predictions
                output = self.model(data)
                probs = torch.exp(output)
                _, preds = torch.max(probs, 1)

                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Update progress
                logger.update_task(test_task)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_targets, all_probs)

        # Generate visualizations
        self.plot_confusion_matrix(all_preds, all_targets)
        self.plot_prediction_distribution(all_probs)

        return metrics

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> dict:
        """Calculate evaluation metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        # Calculate basic metrics
        accuracy = (predictions == targets).mean()

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average='weighted'
        )

        # Calculate per-class metrics
        class_report = classification_report(
            targets,
            predictions,
            target_names=self.reciters,
            output_dict=True
        )

        # Calculate top-k accuracy
        top_k_accuracy = {}
        for k in [1, 3, 5]:
            top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
            top_k_correct = np.any(
                top_k_preds == targets.reshape(-1, 1),
                axis=1
            )
            top_k_accuracy[f'top_{k}_accuracy'] = top_k_correct.mean()

        # Combine metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            **top_k_accuracy,
            'per_class_metrics': class_report
        }

        # Save metrics
        with open(self.eval_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """Plot and save confusion matrix.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
        """
        plt.figure(figsize=(15, 15))
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.reciters,
            yticklabels=self.reciters
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'confusion_matrix.png')
        plt.close()

    def plot_prediction_distribution(self, probabilities: np.ndarray):
        """Plot and save prediction probability distribution.

        Args:
            probabilities: Prediction probabilities
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(probabilities.max(axis=1), bins=50)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Count')
        plt.savefig(self.eval_dir / 'prediction_distribution.png')
        plt.close()

    def analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray
    ):
        """Analyze and save error analysis.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            probabilities: Prediction probabilities
        """
        # Find misclassified samples
        errors = predictions != targets
        error_indices = np.where(errors)[0]

        error_analysis = []
        for idx in error_indices:
            error_analysis.append({
                'true_label': self.reciters[targets[idx]],
                'predicted_label': self.reciters[predictions[idx]],
                'confidence': float(probabilities[idx][predictions[idx]]),
                'top_3_predictions': [
                    {
                        'reciter': self.reciters[i],
                        'probability': float(probabilities[idx][i])
                    }
                    for i in np.argsort(probabilities[idx])[-3:][::-1]
                ]
            })

        # Save error analysis
        with open(self.eval_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=4)


def main():
    """Main evaluation pipeline."""
    with PipelineLogger("evaluate") as logger:
        try:
            # Find latest experiment
            experiments = sorted(
                [d for d in RUNS_DIR.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if not experiments:
                raise ValueError("No experiments found")

            latest_exp = experiments[0]
            logger.log_info(f"Evaluating experiment: {latest_exp.name}")

            # Load model
            logger.log_info("Loading model...")
            model = ReciterCNN(MODEL_CONFIG)
            checkpoint = torch.load(
                latest_exp / 'checkpoints/best_checkpoint.pt',
                map_location='cpu'
            )
            model.load_state_dict(checkpoint['model_state_dict'])

            # Create evaluator
            evaluator = ModelEvaluator(model, latest_exp)

            # Create data loader
            logger.log_info("Loading test data...")
            data_loaders = create_dataloaders(PROCESSED_DATA_DIR)

            # Evaluate model
            logger.log_info("Starting evaluation...")
            metrics = evaluator.evaluate(data_loaders['test'], logger)

            # Log results
            logger.log_stats({
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Top-3 Accuracy': f"{metrics['top_3_accuracy']:.4f}",
                'Top-5 Accuracy': f"{metrics['top_5_accuracy']:.4f}"
            })

            logger.log_success("Evaluation completed successfully")

        except Exception as e:
            logger.log_error(f"Evaluation failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
