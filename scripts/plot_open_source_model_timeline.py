from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates


@dataclass
class ModelInfo:
    name: str
    release_date: datetime
    sizes: list[float]
    model_type: str
    organization: str


def plot_model_timeline(model_infos: list[ModelInfo], save_path: str | None = None):
    # Convert date strings to datetime objects and separate names
    dates = [model_info.release_date for model_info in model_infos]
    names = [
        f"{model_info.name} (MOE)"
        if model_info.model_type == "moe"
        else model_info.name
        for model_info in model_infos
    ]
    sizes = [max(model_info.sizes) for model_info in model_infos]
    organizations = [model_info.organization for model_info in model_infos]

    # Convert dates to matplotlib format
    dates_num = mdates.date2num(dates)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create a color map for organizations
    unique_orgs = list(set(organizations))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_orgs)))
    org_to_color = dict(zip(unique_orgs, colors))
    point_colors = [org_to_color[org] for org in organizations]

    # Plot the dates and sizes
    scatter = ax.scatter(dates_num, sizes, c=point_colors, zorder=5)

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            label=org,
            markersize=8,
        )
        for org, color in org_to_color.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Annotate the points with names
    for date_num, size, name in zip(dates_num, sizes, names):
        ax.text(date_num, size * 1.05, name, ha="center", va="bottom", fontsize=9)

    # Formatting the axes
    ax.set_yscale("log")
    ax.set_ylabel("Largest Model Size (B parameters)")
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Formatting the x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Only show years
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Timeline of Open Source Language Models")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


model_infos = [
    ModelInfo(
        "Llama 3.1",
        datetime(2024, 7, 23),
        [8, 70, 405],
        model_type="dense",
        organization="Meta",
    ),
    ModelInfo(
        "Llama 2",
        datetime(2023, 7, 18),
        [7, 13, 34, 70],
        model_type="dense",
        organization="Meta",
    ),
    ModelInfo(
        "Llama",
        datetime(2023, 2, 27),
        [6.7, 13, 32.5, 65.2],
        model_type="dense",
        organization="Meta",
    ),
    ModelInfo(
        "Mistral Large 2",
        datetime(2024, 7, 24),
        [123],
        model_type="dense",
        organization="Mistral",
    ),
    ModelInfo(
        "Mistral 7B",
        datetime(2023, 9, 27),
        [7],
        model_type="dense",
        organization="Mistral",
    ),
    ModelInfo(
        "Gemma 2",
        datetime(2024, 6, 26),
        [2, 9, 27],
        model_type="dense",
        organization="Google",
    ),
    ModelInfo(
        "Gemma",
        datetime(2024, 2, 21),
        [2, 7],
        model_type="dense",
        organization="Google",
    ),
    ModelInfo(
        "GPT-2",
        datetime(2019, 2, 14),
        [0.117, 0.345, 0.762, 1.5],
        model_type="dense",
        organization="OpenAI",
    ),
    ModelInfo(
        "DeepSeek V3",
        datetime(2024, 12, 26),
        [671],
        model_type="moe",
        organization="DeepSeek",
    ),
    ModelInfo(
        "DeepSeek V2",
        datetime(2024, 5, 7),
        [236],
        model_type="moe",
        organization="DeepSeek",
    ),
    ModelInfo(
        "DeepSeek V1",
        datetime(2024, 1, 5),
        [7, 67],
        model_type="dense",
        organization="DeepSeek",
    ),
]

# Example usage
plot_model_timeline(model_infos, save_path="results/open_source_LLM_timeline.png")
