import math
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import typer

from telco_customer_churn.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


def plot_counts(
    df: pd.DataFrame,
    subplot_size: tuple[int, int] = (5, 4),
    style: str = "whitegrid",
    context: str = "notebook",
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

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The generated figure and axes objects.
    """
    categorical_cols = df.select_dtypes(include="str").columns.to_list()
    sns.set_theme(style=style, context=context)

    n = len(categorical_cols)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    figsize = (subplot_size[0] * n_cols, subplot_size[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, categorical_cols):
        order = df[col].value_counts(dropna=False).index
        sns.countplot(data=df, x=col, order=order, ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Distribution of Categorical Variables", y=1.02)
    plt.tight_layout()

    return fig, axes


if __name__ == "__main__":
    app()
