import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def analyze_correlation(file_path):
    # 1. Load the dataset
    df = pd.read_csv(file_path)

    # 2. Cleanup: Ensure we only use rows where both scores exist
    # (Especially if any German rows failed to match)
    df_clean = df.dropna(subset=['formality_score', 'human_scores'])

    count = len(df_clean)
    print(f"Analyzing {count} examples...")

    # 3. Calculate Correlations
    # Pearson: measures linear relationship
    pearson_val, p_p = pearsonr(df_clean['formality_score'], df_clean['human_scores'])

    # Spearman: measures if the scores move in the same direction (better for style scales)
    spearman_val, p_s = spearmanr(df_clean['formality_score'], df_clean['human_scores'])

    print("\n--- Correlation Results ---")
    print(f"Pearson Correlation:  {pearson_val:.4f} (p-value: {p_p:.4f})")
    print(f"Spearman Correlation: {spearman_val:.4f} (p-value: {p_s:.4f})")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_clean, x='formality_score', y='human_scores',
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

    plt.title(f'LLM vs Human Formality Scores (n={count})', fontsize=15)
    plt.xlabel('LLM Score (formality_score)', fontsize=12)
    plt.ylabel('Human Score (human_scores)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add correlation text to the plot
    plt.text(df_clean['formality_score'].min(), df_clean['human_scores'].max(),
             f'Spearman r: {spearman_val:.2f}',
             bbox=dict(facecolor='white', alpha=0.8))

    # Save and show
    output_image = "correlation_plot.png"
    plt.savefig(output_image)
    print(f"\nPlot saved as: {output_image}")
    plt.show()


if __name__ == "__main__":
    # Ensure this matches your final file name exactly
    filename = 'formality_scores_human&LLM_annotated.csv'
    analyze_correlation(filename)