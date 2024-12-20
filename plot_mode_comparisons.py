import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats

def plot_mode_comparisons():
    plt.style.use('seaborn-v0_8')
    base_dir = Path("experiments/experiment0_baseline/results")
    plots_dir = Path("experiments/plots/baseline_results")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = {}
    for mode in ['gold_only', 'gold_random', 'gold_adversarial']:
        file_path = base_dir / mode / f"results_{mode}_20241219_012159.json"
        if mode == 'gold_random':
            file_path = base_dir / mode / "results_gold_random_20241219_023033.json"
        elif mode == 'gold_adversarial':
            file_path = base_dir / mode / "results_gold_adversarial_20241219_032142.json"
        
        with open(file_path) as f:
            results[mode] = json.load(f)

    # Plot 1: Enhanced Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    accuracies = {mode: np.mean([r['llm_evaluation']['correct'] for r in data]) 
                 for mode, data in results.items()}
    errors = {mode: stats.sem([r['llm_evaluation']['correct'] for r in data]) 
             for mode, data in results.items()}
    
    x = np.arange(len(accuracies))
    bars = ax.bar(x, accuracies.values(), yerr=errors.values(), capsize=5, 
                 color=['#2ecc71', '#e74c3c', '#3498db'])
    
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_xlabel('Experiment Mode', fontsize=12)
    ax.set_title('RAG Performance Across Different Modes\nwith Standard Error', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Gold Only', 'Gold + Random', 'Gold + Adversarial'], rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'enhanced_accuracy_comparison.png', dpi=300)
    plt.close()

    # Plot 2: Enhanced Correct vs Incorrect Distribution
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(results))
    width = 0.35

    for mode_idx, (mode, data) in enumerate(results.items()):
        correct = sum(1 for r in data if r['llm_evaluation']['correct'])
        incorrect = len(data) - correct
        ax.bar(x[mode_idx], correct, width, label='Correct', color='#2ecc71', alpha=0.7)
        ax.bar(x[mode_idx], incorrect, width, bottom=correct, label='Incorrect', color='#e74c3c', alpha=0.7)

    ax.set_ylabel('Number of Responses', fontsize=12)
    ax.set_xlabel('Experiment Mode', fontsize=12)
    ax.set_title('Distribution of Correct and Incorrect Responses\nAcross Different RAG Modes', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Gold Only', 'Gold + Random', 'Gold + Adversarial'], rotation=45)
    
    # Add legend with better positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'enhanced_response_distribution.png', dpi=300)
    plt.close()

    # Plot 3: Enhanced Score Distribution with Violin Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    data_for_violin = []
    labels = []
    
    for mode, data in results.items():
        scores = [r['llm_evaluation']['score'] for r in data]
        data_for_violin.extend(scores)
        labels.extend([mode] * len(scores))
    
    df = pd.DataFrame({
        'Mode': labels,
        'Score': data_for_violin
    })
    
    sns.violinplot(data=df, x='Mode', y='Score', ax=ax, 
                  palette=['#2ecc71', '#e74c3c', '#3498db'])
    
    ax.set_xlabel('Experiment Mode', fontsize=12)
    ax.set_ylabel('LLM Evaluation Score', fontsize=12)
    ax.set_title('Distribution of LLM Evaluation Scores\nAcross Different RAG Modes', fontsize=14, pad=20)
    ax.set_xticklabels(['Gold Only', 'Gold + Random', 'Gold + Adversarial'], rotation=45)
    
    # Add median lines and quartile values
    for i, mode in enumerate(results.keys()):
        scores = [r['llm_evaluation']['score'] for r in results[mode]]
        median = np.median(scores)
        q1, q3 = np.percentile(scores, [25, 75])
        ax.text(i, median, f'Median: {median:.3f}', ha='center', va='bottom')
        ax.text(i, q1, f'Q1: {q1:.3f}', ha='center', va='top')
        ax.text(i, q3, f'Q3: {q3:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'enhanced_score_distribution.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_mode_comparisons()