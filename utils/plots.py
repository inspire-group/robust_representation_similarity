import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict


def heatmap(ax, data, metadata, cmap="inferno"):
    """
    data is in a [N, M] grid shape, where for each value of n (y-axis), we
    have data for all values of m (x-axis). We invert the y-axis to start plotting
    from bottom left (imshow by default plots from top left)
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 2
    fig = ax.imshow(data, cmap=cmap, vmin=metadata.vmin, vmax=metadata.vmax)
    ax.set_xlabel(metadata.xlabel)
    ax.set_ylabel(metadata.ylabel)
    ax.set_xticks(metadata.xticks)
    ax.set_yticks(metadata.yticks)
    ax.set_title(metadata.title, fontsize=11)
    ax.invert_yaxis()
    ax.grid(False)
    return fig


def plot_helper(cka_dir):
    fig, axs = plt.subplots(1, 3, figsize=(3 * 3, 3), sharey=True)
    titles = ["NonRobust-NonRobust", "Robust-Robust", "NonRobust-Robust"]

    for i in range(3):
        if i == 0:
            cka = torch.load(
                os.path.join(
                    cka_dir,
                    "./Model_wrn_28_4_wrn_28_4-Adv_False_False-layers_all_all-reuse_adv_False-identical_models_True-identical_attacks_True-batchsize_1024.pt",
                )
            )["cka"]
        elif i == 1:
            cka = torch.load(
                os.path.join(
                    cka_dir,
                    "./Model_wrn_28_4_wrn_28_4-Adv_False_False-layers_all_all-reuse_adv_False-identical_models_True-identical_attacks_True-batchsize_1024_adv_model.pt",
                )
            )["cka"]
        elif i == 2:
            cka = torch.load(
                os.path.join(
                    cka_dir,
                    "./Model_wrn_28_4_wrn_28_4-Adv_False_False-layers_all_all-reuse_adv_False-identical_models_False-identical_attacks_False-batchsize_1024_cross_model.pt",
                )
            )["cka"]
        else:
            raise ValueError

        metadata = EasyDict(
            {
                "xlabel": "",
                "ylabel": "Layer (Benign features)" if i == 0 else "",
                "xticks": np.arange(0, cka.shape[0], 25),
                "yticks": np.arange(0, cka.shape[1], 25),
                "title": titles[i],
                "vmin": 0.0,
                "vmax": 1.0,
            }
        )
        f = heatmap(axs[i], cka, metadata)

    fig.colorbar(f, ax=axs.ravel().tolist(), shrink=0.72, format="%.1f")
    fig.supxlabel("Layer (Benign features)", x=0.45, y=0.06)
    fig.savefig(
        os.path.join(cka_dir, "./cka_plot.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    plot_helper(cka_dir="./results/")
