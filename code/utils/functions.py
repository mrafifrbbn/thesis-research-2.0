# Function to create parent folder, given a full absolute path as dictionary or string
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.interpolate import griddata
import scipy.optimize as so

from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

def create_parent_folder(full_abspath) -> None:
    if isinstance(full_abspath, dict):
        for filepath in full_abspath.values():
            output_filepath = Path(filepath)
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
    elif isinstance(full_abspath, str):
        output_filepath = Path(full_abspath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

# Linear + parabola for magnitude count
def completeness_linear_parabola(x, beta, x0, b, alpha=0.6):
    a = (alpha - b) / (2 * x0)
    y_pred = np.piecewise(
        x,
        [x <= x0, x > x0],
        [lambda x: alpha * x + beta, lambda x: ((alpha - b) / (2 * x0)) * x**2 + b * x + 0.5 * (alpha - b) * x0 + beta]
    )
    return y_pred

# Only linear (for LAMOST)
def completeness_linear(x, beta, alpha=0.6):
    y_pred = alpha * x + beta
    return y_pred


def find_contour_level(x, proba_xy, volume_pct):
    """Calculates the level of a P(x+Δx, y+Δy) function corresponding to the volume_pct.
    This does it by calculating the total probability (volume or sum) of proba_xy above trial value x,
    and find x where (total probability - volume_pct) crosses from positive to negative, i.e. the root.

    Args:
        x (float): trial value of P(x+Δx, y+Δy) to be computed. 
            If x = 0 (P(x+Δx, y+Δy) = 0), all values > x is the entire data, and so the sum is 1.
            If x = 1 (P(x+Δx, y+Δy) = 0), all values > x is an empty set, and so the sum is 0.
        proba_xy (2D array): 2D values of P(x+Δx, y+Δy).
        volume_pct (float): this is the percentage of data under the mode we want to find the level at.

    Returns:
        level (float): the level corresponding to the volume_pct.
    """
    level = proba_xy[proba_xy > x].sum() - volume_pct
    return level

def density_contour(x, y, bins_levels_tuple_list: List[Tuple[float, Tuple[float, float], Tuple]], ax=None, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """
    for _ in bins_levels_tuple_list:
        # Sigma level to plot and bins along x and y axes
        sigma_level, (bin_size_x, bin_size_y), color_shade = _
        
        # Calculate the normalized count (PDF) inside a square of size given by x_edges and y_edges
        # x_edges and y_edges are the bin edges along the x and y axes
        # pdf, xedges, yedges = np.histogram2d(x, y, bins=(nbins_x, nbins_y), density=True)
        bins_x = np.arange(x.min(), x.max(), bin_size_x)
        bins_y = np.arange(y.min(), y.max(), bin_size_y)
        pdf, xedges, yedges = np.histogram2d(x, y, bins=(bins_x, bins_y), density=True)

        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]
        bin_size = dx * dy

        # By multiplying the PDF with the area, we get the probability at each given area P(x+Δx, y+Δy)
        proba_xy = (pdf * bin_size)

        # # Calculates the level corresponding to the desired sigma-value
        # levels = []
        # for pct_ in pct_lines:
        #     levels.append(so.brentq(find_contour_level, 0., 1., args=(proba_xy, pct_)))
        # levels = sorted(levels)
        levels = so.brentq(find_contour_level, 0., 1., args=(proba_xy, sigma_level))

        # Calculate the bin centers and the corresponding z-value
        X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
        Z = proba_xy.T
        
        # if ax == None:
        #     plt.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        # else:
        #     print(levels)
        ax.contour(X, Y, Z, levels=[levels], origin="lower", colors=[color_shade], **contour_kwargs)