# Function to create parent folder, given a full absolute path as dictionary or string
import os
import sys
import numpy as np
from pathlib import Path

import scipy.optimize as so
from scipy.odr import ODR, Model, RealData

from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create linear function
def linear_func(x: np.array, m: float, c: float):
    return m * x + c

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


# Gaussian function (for fitting)
def gaus(x, mu, sig):
    return (1 / np.sqrt(2 * np.pi * sig**2)) * np.exp(-0.5 * ((x - mu) / sig)**2)

# Skew-normal distribution
def skewnormal(x, loc, err, alpha):
    A = 1 / (np.sqrt(2 * np.pi) * err)
    B = np.exp(-(x - loc)**2/(2 * err**2))
    C = 1 + erf(alpha * (x - loc) / (np.sqrt(2) * err))
    return A * B * C


def remove_outliers(x, k=5.0):
    # Select inliers using MAD
    median_x = np.median(x)
    MAD_x = np.median(np.absolute(x - median_x))
    lower_ = median_x - k * MAD_x
    upper_ = median_x + k * MAD_x
    x_inliers = x[(x >= lower_) & (x <= upper_)]

    # Calculate standard deviation using the assumed inliers
    mean_ = np.mean(x_inliers)
    std_ = np.std(x_inliers)
    lower_ = mean_ - 5 * std_
    upper_ = mean_ + 5 * std_

    # Define outliers as 5 sigma away from the mean
    x_outliers = x[(x < lower_) | (x > upper_)]
    return x_inliers, x_outliers


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


def bin_data(x: np.array, y: np.array, yerr: np.array, xmin: float, xmax: float, n_bin: int):
    # x_bin = np.linspace(np.min(x), np.max(x), n_bin)
    x_bin = np.linspace(xmin, xmax, n_bin)
    x_middle = 0.5 * (x_bin[1:] + x_bin[:-1])
    delta_x = np.diff(x_bin)[0]

    x_bin_ = []
    y_bin = []
    y_bin_err = []
    y_bin_stderr = []

    for x_trial in x_middle:
        x_lower = x_trial - 0.5 * delta_x 
        x_upper = x_trial + 0.5 * delta_x

        y_ = y[(x >= x_lower) & (x < x_upper)]

        if len(y_):
            x_bin_.append(x_trial)
            y_bin.append(np.mean(y_))
            y_bin_err.append(np.std(y_))
            y_bin_stderr.append(np.std(y_) / np.sqrt(len(y_)))
        else:
            continue

    return np.array(x_bin_), np.array(y_bin), np.array(y_bin_err), np.array(y_bin_stderr)


def bin_data_median(x: np.array, y: np.array, yerr: np.array, xmin: float, xmax: float, n_bin: int):
    # x_bin = np.linspace(np.min(x), np.max(x), n_bin)
    x_bin = np.linspace(xmin, xmax, n_bin)
    x_middle = 0.5 * (x_bin[1:] + x_bin[:-1])
    delta_x = np.diff(x_bin)[0]

    x_bin_ = []
    y_bin = []
    y_bin_err = []
    y_bin_stderr = []

    for x_trial in x_middle:
        x_lower = x_trial - 0.5 * delta_x 
        x_upper = x_trial + 0.5 * delta_x

        y_ = y[(x >= x_lower) & (x < x_upper)]

        if len(y_):
            x_bin_.append(x_trial)
            y_bin.append(np.median(y_))
            y_bin_err.append(np.std(y_))
            y_bin_stderr.append(np.std(y_) / np.sqrt(len(y_)))
        else:
            continue

    return np.array(x_bin_), np.array(y_bin), np.array(y_bin_err), np.array(y_bin_stderr)


def bin_data_error_weighting(x: np.array, y: np.array, yerr: np.array, xmin: float, xmax: float, n_bin: int):
    # x_bin = np.linspace(np.min(x), np.max(x), n_bin)
    x_bin = np.linspace(xmin, xmax, n_bin)
    x_middle = 0.5 * (x_bin[1:] + x_bin[:-1])
    delta_x = np.diff(x_bin)[0]

    x_bin_ = []
    y_bin = []
    y_bin_err = []
    y_bin_stderr = []

    for x_trial in x_middle:
        x_lower = x_trial - 0.5 * delta_x 
        x_upper = x_trial + 0.5 * delta_x

        mask_ = (x >= x_lower) & (x < x_upper)

        y_ = y[mask_]
        yerr_ = yerr[mask_]

        if len(y_) >= 2:
            # Calculate error-weighted mean
            w = 1 / yerr_**2
            w_sum = np.sum(w)
            y_mean = np.sum(w * y_) / w_sum
            y_stderr = 1 / np.sqrt(w_sum)

            x_bin_.append(x_trial)
            y_bin.append(y_mean)
            y_bin_err.append(np.std(y_))
            y_bin_stderr.append(y_stderr)
        else:
            continue

    return np.array(x_bin_), np.array(y_bin), np.array(y_bin_err), np.array(y_bin_stderr)


def ODR_linear_fit(x, y, xerr=None, yerr=None, m_guess=1.0, b_guess=0.0, left_boundary=None, right_boundary=None):
    """Helper function to fit a line with errors in both x and y axes.

    Args:
        x (): x data
        y (_type_): y data
        xerr (_type_, optional): error in x. Defaults to None.
        yerr (_type_, optional): error in y. Defaults to None.
        m_guess (float, optional): initial guess for the slope. Defaults to 1.0.
        b_guess (float, optional): initial guess for the intercept. Defaults to 0.0.
        left_boundary (_type_, optional): starting point for prediction line. Defaults to None.
        right_boundary (_type_, optional): ending point for prediction line. Defaults to None.
    """
    def f(B, x):
        '''Linear function y = m*x + b'''
        # B is a vector of the parameters.
        # x is an array of the current x values.
        # x is in the same format as the x passed to Data or RealData.
        #
        # Return an array in the same format as y passed to Data or RealData.
        return B[0]*x + B[1]
    
    # ODR stuff
    linear = Model(f)
    mydata = RealData(x=x, y=y, sx=xerr, sy=yerr)
    odr = ODR(mydata, linear, beta0=[m_guess, b_guess])
    odr_output = odr.run() 
    m_pred, b_pred = odr_output.beta
    m_err, b_err = np.sqrt(np.diag(odr_output.cov_beta))
    print(f"Slope: {m_pred} ± {m_err} | Intercept: {b_pred} ± {b_err}")

    # Create MC sample
    n_trial = 10000
    m_trial, b_trial = np.random.multivariate_normal(odr_output.beta, odr_output.cov_beta, n_trial).T

    # Plot boundaries
    left_ = x.min()
    right_ = x.max()

    if left_boundary:
        left_ = left_boundary
    if right_boundary:
        right_ = right_boundary

    x_pred = np.linspace(left_, right_, 1000)
    y_trial = m_trial.reshape(-1, 1) * x_pred + b_trial.reshape(-1, 1)
    y_pred = m_pred * x_pred + b_pred
    y_pred_lower = np.quantile(y_trial, q=0.16, axis=0)
    y_pred_upper = np.quantile(y_trial, q=0.84, axis=0)

    return odr_output, x_pred, y_pred, y_pred_lower, y_pred_upper