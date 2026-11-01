"""Simple matplotlib drawings; no GUI framework is needed."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent / 'outputs'


def draw(ax, original, transformed, title, T=None):
    for points, color, style, label in [(original, 'blue', '--', 'Original'),
                                        (transformed, 'red', '-', 'Transformed')]:
        closed = np.column_stack((points, points[:,0]))
        ax.plot(closed[0], closed[1], style, color=color, marker='o', label=label)
    ax.fill(transformed[0], transformed[1], color='red', alpha=0.2)
    if T is not None:
        title += '\nT = ' + np.array2string(T, precision=2) + f'\ndet(T) = {np.linalg.det(T):.3g}'
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(fontsize=8)


def finish(fig, filename=None):
    fig.tight_layout()
    if filename:
        OUTPUTS.mkdir(exist_ok=True)
        path = OUTPUTS / filename
        fig.savefig(path, dpi=140)
        print('Saved plot:', path.name)
    else:
        plt.show()
    plt.close(fig)


def plot_shape(original, transformed, title, T=None, filename=None):
    fig, ax = plt.subplots(figsize=(7,6))
    draw(ax, original, transformed, title, T)
    finish(fig, filename)


def chain(shape, A, B, filename=None):
    # AB applies B first; BA applies A first (column-vector convention).
    AB, BA = A @ B, B @ A
    print('AB:\n', AB, '\nBA:\n', BA)
    print('AB equals BA:', np.allclose(AB, BA))
    print('Order can matter; some special pairs do commute.')
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    draw(axes[0], shape, AB @ shape, 'B first, then A: AB', AB)
    draw(axes[1], shape, BA @ shape, 'A first, then B: BA', BA)
    finish(fig, filename)
    return AB @ shape, BA @ shape
