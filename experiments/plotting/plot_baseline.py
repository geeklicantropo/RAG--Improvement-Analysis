import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import logging
from tqdm import tqdm

class BaselinePlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "baseline"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        plt.style.use('ggplot')
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("BaselinePlotter")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        fh = logging.FileHandler(self.plots_dir / 'plotting.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def _plot_document_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        try:
            if 'document_positions' in df.columns:
                position_data = df['document_positions'].explode()
                sns.histplot(data=position_data, discrete=True)
                plt.title('Document Position Distribution')
                plt.xlabel('Document Position')
                plt.ylabel('Count')
                plt.savefig(self.plots_dir / 'document_distribution.png', dpi=300)
        finally:
            plt.close()

    def _plot_retriever_comparison(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 6))
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            retriever_accuracy = df.groupby('retriever_type')['accuracy'].agg(['mean', 'std']).reset_index()
            sns.barplot(data=retriever_accuracy, x='retriever_type', y='mean', yerr=retriever_accuracy['std'], ax=ax1)
            ax1.set_title('Retriever Accuracy Comparison')
            
            # Response time comparison
            if 'response_time' in df.columns:
                sns.boxplot(data=df, x='retriever_type', y='response_time', ax=ax2)
                ax2.set_title('Response Time by Retriever')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'retriever_comparison.png', dpi=300)
        finally:
            plt.close()

    def _plot_noise_impact_analysis(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 10))
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Noise impact on accuracy
            if all(col in df.columns for col in ['noise_ratio', 'accuracy', 'retriever_type']):
                sns.lineplot(data=df, x='noise_ratio', y='accuracy', hue='retriever_type', ax=axes[0,0])
                axes[0,0].set_title('Noise Impact on Accuracy')
                
            # Noise distribution by retriever
            if 'noise_type' in df.columns:
                sns.countplot(data=df, x='noise_type', hue='retriever_type', ax=axes[0,1])
                axes[0,1].set_title('Noise Distribution by Retriever')
                
            # Performance under noise
            if 'response_time' in df.columns:
                sns.scatterplot(data=df, x='noise_ratio', y='response_time', hue='retriever_type', ax=axes[1,0])
                axes[1,0].set_title('Performance Under Noise')
                
            # Document relevance with noise
            if 'relevance_score' in df.columns:
                sns.boxplot(data=df, x='noise_type', y='relevance_score', ax=axes[1,1])
                axes[1,1].set_title('Document Relevance with Noise')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'noise_impact_analysis.png', dpi=300)
        finally:
            plt.close()

    def plot_results(self, results: List[Dict]):
        if not results:
            self.logger.warning("No results to plot")
            return
            
        self.logger.info("Processing baseline results for visualization")
        clean_results = [r for r in results if r.get('success', False)]
        metrics_df = pd.DataFrame([r['metrics'] for r in clean_results])
        
        if metrics_df.empty:
            self.logger.warning("No valid metrics found")
            return

        plot_functions = [
            self._plot_retrieval_comparison,
            self._plot_noise_analysis,
            self._plot_generation_metrics,
            self._plot_noise_impact_by_retriever,
            self._plot_performance_trends,
            self._plot_document_distribution,
            self._plot_retriever_comparison,
            self._plot_noise_impact_analysis
        ]

        for plot_func in tqdm(plot_functions, desc="Generating plots"):
            try:
                plot_func(metrics_df)
            except Exception as e:
                self.logger.error(f"Error in {plot_func.__name__}: {str(e)}")

    def _plot_retrieval_comparison(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        try:
            retriever_metrics = ['clean_accuracy', 'noisy_accuracy']
            retriever_data = []
            
            for retriever in ['contriever', 'bm25']:
                for metric in retriever_metrics:
                    if f"{retriever}_{metric}" in df.columns:
                        retriever_data.append({
                            'Retriever': retriever,
                            'Type': 'Clean' if 'clean' in metric else 'Noisy',
                            'Accuracy': df[f"{retriever}_{metric}"].mean()
                        })
            
            if retriever_data:
                plot_df = pd.DataFrame(retriever_data)
                sns.barplot(data=plot_df, x='Retriever', y='Accuracy', hue='Type')
                plt.title('Retrieval Accuracy Comparison')
                plt.savefig(self.plots_dir / 'retrieval_comparison.png', dpi=300)

        except Exception as e:
            self.logger.error(f"Error in retrieval comparison: {str(e)}")
        finally:
            plt.close()

    def _plot_noise_analysis(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 6))
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Noise distribution
            if 'noise_ratio' in df.columns:
                sns.histplot(data=df, x='noise_ratio', ax=ax1)
                ax1.set_title('Noise Ratio Distribution')
                
            # Accuracy vs Noise
            if all(col in df.columns for col in ['noise_ratio', 'accuracy']):
                sns.regplot(data=df, x='noise_ratio', y='accuracy', ax=ax2)
                ax2.set_title('Accuracy vs Noise Ratio')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'noise_analysis.png', dpi=300)
            
        except Exception as e:
            self.logger.error(f"Error in noise analysis: {str(e)}")
        finally:
            plt.close()

    def _plot_generation_metrics(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 8))
        try:
            metrics = ['generation_accuracy', 'retrieval_accuracy', 'response_length']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))
                for i, metric in enumerate(available_metrics):
                    sns.boxplot(data=df, y=metric, ax=axes[i])
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'generation_metrics.png', dpi=300)
                
        except Exception as e:
            self.logger.error(f"Error in generation metrics: {str(e)}")
        finally:
            plt.close()

    def _plot_noise_impact_by_retriever(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        try:
            if all(col in df.columns for col in ['noise_ratio', 'retriever_type', 'accuracy']):
                sns.lineplot(
                    data=df,
                    x='noise_ratio',
                    y='accuracy',
                    hue='retriever_type',
                    style='retriever_type',
                    markers=True
                )
                plt.title('Noise Impact by Retriever Type')
                plt.savefig(self.plots_dir / 'noise_impact_by_retriever.png', dpi=300)
                
        except Exception as e:
            self.logger.error(f"Error in noise impact by retriever: {str(e)}")
        finally:
            plt.close()

    def _plot_performance_trends(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 6))
        try:
            performance_metrics = ['response_time', 'memory_usage', 'context_length']
            available_metrics = [m for m in performance_metrics if m in df.columns]
            
            if available_metrics and 'noise_ratio' in df.columns:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))
                
                for i, metric in enumerate(available_metrics):
                    sns.regplot(
                        data=df,
                        x='noise_ratio',
                        y=metric,
                        ax=axes[i]
                    )
                    axes[i].set_title(f'{metric.replace("_", " ").title()} vs Noise')
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'performance_trends.png', dpi=300)
                
        except Exception as e:
            self.logger.error(f"Error in performance trends: {str(e)}")
        finally:
            plt.close()