import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sns.set_theme()


def plot_prediction(x, y, model, pipeline, ax=None):
    if ax is None:
        ax = plt.gca()

    torch_x = torch.from_numpy(x).unsqueeze(0)

    with torch.no_grad():
        predicted = model(torch_x)

    close_mean, close_std = (
        pipeline.named_steps["scale"]["scale"].mean_[0],
        pipeline.named_steps["scale"]["scale"].scale_[0],
    )

    input_data = x[:, 0] * close_std + close_mean
    actual = y * close_std + close_mean
    predicted = predicted.item() * close_std + close_mean

    sns.lineplot(x=np.arange(x.shape[0]), y=input_data, ax=ax)
    sns.scatterplot(x=[x.shape[0]], y=actual, ax=ax)
    sns.scatterplot(x=[x.shape[0]], y=predicted, ax=ax)
