import io

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def to_image(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def plot_samples(noises, samples):
    """
    Plot the noises in black and samples in blue.

    Parameters
    ----------
    noises : Array, shape (bs, 2)
        represents the source minibatch
    samples : Tensor, shape (bs, 2)
        represents the target minibatch
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(noises[:, 0], noises[:, 1], s=10, alpha=0.8, c="black")
    plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=1, c="blue")

    for i in range(len(noises)):
        plt.plot([noises[i, 0], samples[i, 0]],
                 [noises[i, 1], samples[i, 1]],
                 color="gray", linewidth=0.5, alpha=0.5)
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])

    plt.xticks([])
    plt.yticks([])
    return to_image(plt)


def plot_images_grid(
    images: np.ndarray,
    num_rows: int = 4,
    num_cols: int = 8,
    cmap: str = "gray",
    squeeze_channel: bool = True,
):
    """
    Plot a grid of images using matplotlib and return it as a PIL image.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W) or (N, C, H, W), with values in [0, 1] or [-1, 1].
    num_rows : int
        Number of rows in the grid.
    num_cols : int
        Number of columns in the grid.
    cmap : str
        Colormap used for imshow (e.g., 'gray', 'viridis').
    squeeze_channel : bool
        Whether to squeeze singleton channels (i.e., shape (1, H, W) → (H, W)).

    Returns
    -------
    PIL.Image
        Rendered image grid.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Rescale to [0, 1] if in [-1, 1]
    if images.min() < 0:
        images = (images + 1) / 2

    N = images.shape[0]
    plt.figure(figsize=(num_cols, num_rows))
    for i in range(min(N, num_rows * num_cols)):
        plt.subplot(num_rows, num_cols, i + 1)

        img = images[i]
        if img.ndim == 3 and squeeze_channel and img.shape[0] == 1:
            img = img[0]  # (1, H, W) → (H, W)
        elif img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)

        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.axis("off")

    return to_image(plt)

