"""
Performance Dashboard for RAG Technique Comparison
Visualizes BEIR benchmark results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import json

sns.set_style("whitegrid")
sns.set_palette("husl")


class RAGPerformanceDashboard:
    """
    Create visualizations for RAG technique comparison
    """

    def __init__(self, results_df: pd.DataFrame, report: Dict = None):
        """
        Initialize dashboard

        Args:
            results_df: DataFrame with evaluation results
            report: Report dictionary from BEIR evaluation
        """
        self.results_df = results_df
        self.report = report or {}

    def plot_technique_comparison(
        self,
        metric: str = "NDCG@10",
        save_path: str = None
    ):
        """
        Bar chart comparing techniques across datasets

        Args:
            metric: Metric to compare
            save_path: Path to save plot
        """
        plt.figure(figsize=(14, 6))

        # Pivot for plotting
        pivot_df = self.results_df.pivot(
            index="technique",
            columns="dataset",
            values=metric
        )

        pivot_df.plot(kind="bar", figsize=(14, 6))

        plt.title(f"RAG Technique Comparison - {metric}", fontsize=16, fontweight="bold")
        plt.xlabel("Technique", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()

    def plot_metric_heatmap(
        self,
        metrics: List[str] = None,
        save_path: str = None
    ):
        """
        Heatmap showing all metrics for all techniques

        Args:
            metrics: List of metrics to include
            save_path: Path to save plot
        """
        if metrics is None:
            metrics = ["NDCG@10", "MAP@10", "Recall@10", "MRR@10"]

        # Filter to selected metrics
        available_metrics = [m for m in metrics if m in self.results_df.columns]

        # Average across datasets
        avg_df = self.results_df.groupby("technique")[available_metrics].mean()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "Score"}
        )

        plt.title("Average Performance Across All Datasets", fontsize=16, fontweight="bold")
        plt.xlabel("Metric", fontsize=12)
        plt.ylabel("Technique", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()

    def plot_speed_vs_quality(
        self,
        quality_metric: str = "NDCG@10",
        save_path: str = None
    ):
        """
        Scatter plot: execution time vs quality

        Args:
            quality_metric: Quality metric to use
            save_path: Path to save plot
        """
        # Average by technique
        avg_df = self.results_df.groupby("technique").agg({
            quality_metric: "mean",
            "execution_time": "mean"
        }).reset_index()

        plt.figure(figsize=(10, 6))

        plt.scatter(
            avg_df["execution_time"],
            avg_df[quality_metric],
            s=200,
            alpha=0.6,
            edgecolors="black"
        )

        # Annotate points
        for idx, row in avg_df.iterrows():
            plt.annotate(
                row["technique"],
                (row["execution_time"], row[quality_metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        plt.xlabel("Execution Time (seconds)", fontsize=12)
        plt.ylabel(quality_metric, fontsize=12)
        plt.title(
            f"Speed vs Quality Tradeoff",
            fontsize=16,
            fontweight="bold"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()

    def plot_dataset_difficulty(
        self,
        metric: str = "NDCG@10",
        save_path: str = None
    ):
        """
        Compare dataset difficulty (average scores)

        Args:
            metric: Metric to analyze
            save_path: Path to save plot
        """
        # Average score per dataset
        dataset_avg = self.results_df.groupby("dataset")[metric].mean().sort_values()

        plt.figure(figsize=(10, 6))
        dataset_avg.plot(kind="barh", color="steelblue")

        plt.xlabel(f"Average {metric}", fontsize=12)
        plt.ylabel("Dataset", fontsize=12)
        plt.title("Dataset Difficulty Ranking", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()

    def plot_k_value_analysis(
        self,
        technique: str,
        dataset: str,
        save_path: str = None
    ):
        """
        Analyze how metrics change with k

        Args:
            technique: Technique to analyze
            dataset: Dataset to analyze
            save_path: Path to save plot
        """
        # Filter to specific technique and dataset
        subset = self.results_df[
            (self.results_df["technique"] == technique) &
            (self.results_df["dataset"] == dataset)
        ]

        if len(subset) == 0:
            print(f"No data for {technique} on {dataset}")
            return

        # Extract metrics at different k values
        metrics_at_k = {}
        for col in subset.columns:
            if "@" in col:
                metric_name, k_val = col.split("@")
                if metric_name not in metrics_at_k:
                    metrics_at_k[metric_name] = {}
                metrics_at_k[metric_name][int(k_val)] = subset[col].values[0]

        # Plot
        plt.figure(figsize=(10, 6))

        for metric_name, k_values_dict in metrics_at_k.items():
            k_values = sorted(k_values_dict.keys())
            scores = [k_values_dict[k] for k in k_values]

            plt.plot(k_values, scores, marker="o", label=metric_name, linewidth=2)

        plt.xlabel("k (number of documents)", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title(
            f"Metrics vs k - {technique} on {dataset}",
            fontsize=16,
            fontweight="bold"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()

    def plot_ablation_study(
        self,
        ablation_results: Dict[str, float],
        save_path: str = None
    ):
        """
        Visualize ablation study results

        Args:
            ablation_results: {configuration: score}
            save_path: Path to save plot
        """
        configs = list(ablation_results.keys())
        scores = list(ablation_results.values())

        # Calculate improvements
        baseline_score = scores[0]
        improvements = [(s - baseline_score) / baseline_score * 100 for s in scores]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute scores
        ax1.barh(configs, scores, color="steelblue")
        ax1.set_xlabel("nDCG@10", fontsize=12)
        ax1.set_title("Absolute Scores", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="x")

        # Relative improvements
        colors = ["gray" if i == 0 else "green" for i in improvements]
        ax2.barh(configs, improvements, color=colors)
        ax2.set_xlabel("Improvement over Baseline (%)", fontsize=12)
        ax2.set_title("Relative Improvements", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()

    def generate_full_report(self, output_dir: str = "./dashboard_output"):
        """
        Generate complete dashboard with all visualizations

        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating full performance dashboard...")

        # 1. Technique comparison
        print("  - Technique comparison...")
        self.plot_technique_comparison(
            metric="NDCG@10",
            save_path=str(output_path / "technique_comparison.png")
        )

        # 2. Metric heatmap
        print("  - Metric heatmap...")
        self.plot_metric_heatmap(
            save_path=str(output_path / "metric_heatmap.png")
        )

        # 3. Speed vs quality
        print("  - Speed vs quality...")
        self.plot_speed_vs_quality(
            save_path=str(output_path / "speed_vs_quality.png")
        )

        # 4. Dataset difficulty
        print("  - Dataset difficulty...")
        self.plot_dataset_difficulty(
            save_path=str(output_path / "dataset_difficulty.png")
        )

        # 5. Generate summary table
        print("  - Summary table...")
        self._generate_summary_table(output_path)

        print(f"\nDashboard generated: {output_path}")

    def _generate_summary_table(self, output_path: Path):
        """Generate summary table as HTML"""
        # Average across datasets
        summary = self.results_df.groupby("technique").agg({
            "NDCG@10": "mean",
            "MAP@10": "mean",
            "Recall@10": "mean",
            "MRR@10": "mean",
            "execution_time": "mean"
        }).round(4)

        # Rank techniques
        summary["Rank"] = summary["NDCG@10"].rank(ascending=False).astype(int)
        summary = summary.sort_values("Rank")

        # Save as HTML
        html_path = output_path / "summary_table.html"

        html = f"""
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>RAG Technique Performance Summary</h2>
            {summary.to_html()}
        </body>
        </html>
        """

        with open(html_path, "w") as f:
            f.write(html)

        print(f"    Summary table saved: {html_path}")


# Example usage
if __name__ == "__main__":
    # Load sample results
    sample_data = {
        "technique": ["baseline", "cross_encoder", "rag_fusion", "hybrid"] * 3,
        "dataset": ["nfcorpus"] * 4 + ["scifact"] * 4 + ["fiqa"] * 4,
        "NDCG@10": [0.42, 0.56, 0.52, 0.63, 0.38, 0.51, 0.48, 0.58, 0.45, 0.59, 0.54, 0.65],
        "MAP@10": [0.35, 0.48, 0.44, 0.55, 0.32, 0.45, 0.41, 0.51, 0.38, 0.52, 0.47, 0.58],
        "Recall@10": [0.68, 0.75, 0.72, 0.80, 0.62, 0.71, 0.68, 0.76, 0.65, 0.73, 0.70, 0.78],
        "MRR@10": [0.48, 0.61, 0.56, 0.68, 0.42, 0.56, 0.52, 0.63, 0.50, 0.64, 0.59, 0.70],
        "execution_time": [0.05, 0.25, 0.18, 0.32, 0.04, 0.22, 0.17, 0.30, 0.06, 0.28, 0.20, 0.35],
    }

    results_df = pd.DataFrame(sample_data)

    # Create dashboard
    dashboard = RAGPerformanceDashboard(results_df)

    # Generate visualizations
    dashboard.generate_full_report(output_dir="./sample_dashboard")

    # Ablation study
    ablation_data = {
        "Baseline": 0.42,
        "+ Multi-Query": 0.48,
        "+ Reranking": 0.56,
        "+ Fusion": 0.63,
    }

    dashboard.plot_ablation_study(ablation_data, save_path="./sample_dashboard/ablation.png")
