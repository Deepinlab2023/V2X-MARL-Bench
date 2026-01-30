from __future__ import annotations
from typing import Sequence, Optional
import matplotlib.pyplot as plt


def plot_test_returns(
    test_returns: Sequence[float],
    *,
    title: str,
    figure_id: int = 1,
    label: str = "Test Reward",
    xlabel: str = "Test Interval",
    ylabel: str = "Return",
    pause: float = 1.0,
    clear: bool = True,
    grid: bool = True,
) -> None:

    plt.ion()

    plt.figure(figure_id)
    if clear:
        plt.clf()

    plt.plot(test_returns, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if label:
        plt.legend()

    if grid:
        plt.grid(True)

    plt.draw()
    plt.pause(pause)
