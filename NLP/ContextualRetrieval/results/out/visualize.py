import pandas as pd
import plotly.express as px
from pathlib import Path

# Configuration
RESULTS_FILE = Path("results/enhanced_evaluation_metrics.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def visualize_metrics(df: pd.DataFrame, metric: str):
    """
    Visualize a specified metric using box plots for different collections and search types.
    """
    fig = px.box(
        df,
        x="Collection",
        y=f"Avg {metric}",
        color="Search Type",
        title=f"{metric} Distribution by Collection and Search Type",
        labels={"Collection": "Collection Type", f"Avg {metric}": metric}
    )
    fig.write_html(OUTPUT_DIR / f"{metric}_distribution.html")
    print(f"Visualization saved to {OUTPUT_DIR / f'{metric}_distribution.html'}")

def visualize_all_metrics(df: pd.DataFrame):
    """
    Visualize all key metrics from the evaluation results.
    """
    metrics = [
        "precision",
        "recall",
        "mrr",
        "ndcg",
        "max_similarity",
        "mean_similarity",
        "question_type_match"
    ]
    for metric in metrics:
        visualize_metrics(df, metric)

def main():
    """
    Main function to load data and generate visualizations.
    """
    try:
        df = pd.read_csv(RESULTS_FILE)
        visualize_all_metrics(df)
        print("All visualizations generated successfully!")
    except FileNotFoundError:
        print(f"Error: {RESULTS_FILE} not found. Make sure to run evaluate_rag.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()