import io

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
