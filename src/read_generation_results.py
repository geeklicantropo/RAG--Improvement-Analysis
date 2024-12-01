import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from experiment_logger import ExperimentLogger


class ResultsReader:
    """
    Reads, evaluates, and compares results across experiments.
    """

    def __init__(self, results_dir: str, output_dir: str, logger: Optional[ExperimentLogger] = None):
        """
        Initialize the ResultsReader.

        Args:
            results_dir (str): Path to the directory containing experiment results.
            output_dir (str): Directory to save evaluation metrics and comparative results.
            logger (Optional[ExperimentLogger]): Logger for the evaluation process.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or ExperimentLogger("read_generation_results", "logs")
        self.metrics = {}

    def _load_results(self, experiment_type: str) -> List[Dict[str, Any]]:
        """
        Load results for a specific experiment type.

        Args:
            experiment_type (str): Type of the experiment (e.g., 'baseline', 'clustering', 'noise-injected').

        Returns:
            List[Dict[str, Any]]: List of experiment results.
        """
        try:
            self.logger.log_step_start(f"Loading results for {experiment_type}")

            results_path = self.results_dir / experiment_type / "results.json"
            if not results_path.exists():
                self.logger.experiment_logger.warning(f"Results not found for {experiment_type}")
                return []

            with open(results_path, "r") as f:
                results = json.load(f)

            self.logger.log_metric(f"{experiment_type}_results_count", len(results))
            self.logger.log_step_end(f"Results loaded for {experiment_type}")
            return results

        except Exception as e:
            self.logger.log_error(e, f"Error loading results for {experiment_type}")
            raise

    def evaluate_experiment(self, experiment_type: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate results for a specific experiment.

        Args:
            experiment_type (str): Type of the experiment.
            results (List[Dict[str, Any]]): Experiment results.

        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        try:
            self.logger.log_step_start(f"Evaluating {experiment_type}")

            if not results:
                self.logger.experiment_logger.warning(f"No results to evaluate for {experiment_type}")
                return {}

            # Extract predicted answers and gold answers
            predicted_answers = [r["generated_answer"] for r in results]
            gold_answers = [r["gold_answer"] for r in results]

            # Calculate evaluation metrics
            metrics = {
                "accuracy": accuracy_score(gold_answers, predicted_answers),
                "f1_score": f1_score(gold_answers, predicted_answers, average="weighted"),
                "precision": precision_score(gold_answers, predicted_answers, average="weighted"),
                "recall": recall_score(gold_answers, predicted_answers, average="weighted"),
                "total_examples": len(results),
            }

            # Log metrics
            self.logger.log_metrics(metrics)
            self.logger.log_step_end(f"Evaluation complete for {experiment_type}")
            return metrics

        except Exception as e:
            self.logger.log_error(e, f"Error evaluating {experiment_type}")
            raise

    def save_evaluation_metrics(self, experiment_type: str, metrics: Dict[str, Any]):
        """
        Save evaluation metrics for a specific experiment.

        Args:
            experiment_type (str): Type of the experiment.
            metrics (Dict[str, Any]): Evaluation metrics to save.
        """
        try:
            metrics_path = self.output_dir / f"{experiment_type}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            self.logger.experiment_logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            self.logger.log_error(e, f"Error saving metrics for {experiment_type}")
            raise

    def compare_experiments(self, experiment_types: List[str]):
        """
        Compare metrics across multiple experiments.

        Args:
            experiment_types (List[str]): List of experiment types to compare.
        """
        try:
            self.logger.log_step_start("Comparing experiments")

            comparative_metrics = []

            for experiment_type in experiment_types:
                results = self._load_results(experiment_type)
                metrics = self.evaluate_experiment(experiment_type, results)
                metrics["experiment_type"] = experiment_type
                comparative_metrics.append(metrics)

            # Save comparative metrics
            comparative_metrics_df = pd.DataFrame(comparative_metrics)
            comparative_metrics_path = self.output_dir / "comparative_metrics.csv"
            comparative_metrics_df.to_csv(comparative_metrics_path, index=False)

            self.logger.log_step_end("Comparison complete")
            self.logger.experiment_logger.info(f"Saved comparative metrics to {comparative_metrics_path}")

        except Exception as e:
            self.logger.log_error(e, "Error comparing experiments")
            raise

    def run(self, experiment_types: List[str]):
        """
        Execute the full results reading, evaluation, and comparison workflow.

        Args:
            experiment_types (List[str]): List of experiment types to process.
        """
        try:
            self.logger.log_step_start("Processing results")

            for experiment_type in tqdm(experiment_types, desc="Processing experiments"):
                results = self._load_results(experiment_type)
                metrics = self.evaluate_experiment(experiment_type, results)
                self.metrics[experiment_type] = metrics
                self.save_evaluation_metrics(experiment_type, metrics)

            # Compare experiments
            self.compare_experiments(experiment_types)

            self.logger.log_step_end("Results processing complete")

        except Exception as e:
            self.logger.log_error(e, "Error processing results")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read and compare generation results.")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing experiment results."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save evaluation and comparison results."
    )
    parser.add_argument(
        "--experiment_types",
        nargs="+",
        default=["baseline", "clustering", "noise-injected"],
        help="List of experiment types to process."
    )

    args = parser.parse_args()

    reader = ResultsReader(args.results_dir, args.output_dir)
    reader.run(args.experiment_types)
