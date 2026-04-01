import math
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from telco_customer_churn.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def counts(
    input_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_path: Path = FIGURES_DIR / "counts.png",
    subplot_size: tuple[int, int] = (5, 4),
    style: str = "whitegrid",
    context: str = "notebook",
    max_cardinality: int = 20,
) -> None:
    """Generate count plots for all categorical columns in the dataset and save the figure.

    Parameters
    ----------
    input_path : Path, optional
        Path to the input CSV file containing the dataset, by default PROCESSED_DATA_DIR / "features.csv"
    output_path : Path, optional
        Path to save the generated plot, by default FIGURES_DIR / "counts.png"
    subplot_size : tuple[int, int], optional
        Size of each subplot, by default (5, 4)
    style : str, optional
        Style of the plot, by default "whitegrid"
    context : str, optional
        Context of the plot. This is passed to `sns.set_theme()`, by default "notebook"
    max_cardinality : int, optional
        Maximum number of unique values for a categorical column to be included in the count plots, by default 20
    """
    logger.info(f"Generating count plots for categorical columns from {input_path}...")
    df = pd.read_csv(input_path)
    fig, _ = plot_counts(
        df,
        subplot_size=subplot_size,
        style=style,
        context=context,
        max_cardinality=max_cardinality,
    )
    fig.savefig(output_path)
    logger.success(f"Count plots saved to {output_path}.")


def plot_counts(
    df: pd.DataFrame,
    subplot_size: tuple[int, int] = (5, 4),
    style: str = "whitegrid",
    context: str = "notebook",
    max_cardinality: int = 20,
) -> tuple[plt.Figure, plt.Axes]:
    """Generate count plots for all categorical columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be plotted.
    subplot_size : tuple[int, int], optional
        Size of each subplot, by default (5, 4)
    style : str, optional
        Style of the plot, by default "whitegrid"
    context : str, optional
        Context of the plot. This is passed to `sns.set_theme()`, by default "notebook"
    max_cardinality : int, optional
        Maximum number of unique values for a categorical column to be included in the count plots, by default 20

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The generated figure and axes objects.
    """
    candidates = df.select_dtypes(include="str").columns.to_list()
    categoricals = candidates.copy()  # some cols will be filtered out

    for col in candidates:
        if df[col].nunique() > max_cardinality:
            logger.warning(
                f"Column '{col}' has high cardinality ({df[col].nunique()} unique "
                f"values which is greater than max_cardinality={max_cardinality}). "
                "Skipping this column for count plots."
            )
            categoricals.remove(col)

    sns.set_theme(style=style, context=context)

    n = len(categoricals)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    figsize = (subplot_size[0] * n_cols, subplot_size[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, categoricals):
        order = df[col].value_counts(dropna=False).index
        sns.countplot(data=df, x=col, order=order, ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Count Plots of Categorical Columns", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig, axes


if __name__ == "__main__":
    app()
