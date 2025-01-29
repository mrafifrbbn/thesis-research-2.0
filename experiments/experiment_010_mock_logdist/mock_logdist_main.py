import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from src.utils.constants import *
from src.utils.CosmoFunc import *

def fit_logdist(
        df: pd.DataFrame,
        smin: float,
        FPparams: np.ndarray,
        use_full_fn: bool = True,
        mag_high: float = MAG_HIGH,
        mag_low: float = MAG_LOW,
    ) -> pd.DataFrame:
    """
    This is a function to calculate the log-distance ratio posteriors and obtain summary statistics.
    Summary statistics are obtained using two methods: direct calculation assuming skew-normal and 
    fitting using scipy's curve_fit assuming Gaussian.
    """

    # Get some redshift-distance lookup tables
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    dz_cluster = sp.interpolate.splev(df["z"], dist_spline)

    # Define the range of logdists values to be calculated
    dmin, dmax, nd = -1.5, 1.5, 1001
    dbins = np.linspace(dmin, dmax, nd, endpoint=True)

    # Calculate full FN
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
    
    df['r_true'] = df['r']

    # Calculate log-likelihood
    loglike = FP_func(FPparams, dbins, df["z"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["dr"].to_numpy(), df["ds"].to_numpy(), df["di"].to_numpy(), np.ones(len(df)), smin, lmin, lmax, df["C_m"].to_numpy(), sumgals=False, use_full_fn=use_full_fn)

    # Calculate full FN
    FNvals = FN_func(FPparams, df["z"].to_numpy(), df["dr"].to_numpy(), df["ds"].to_numpy(), df["di"].to_numpy(), lmin, lmax, smin)

    # Convert to the PDF for logdistance
    logP_dist = -1.5 * np.log(2.0 * math.pi) - loglike - FNvals

    # normalise logP_dist (trapezoidal rule)
    ddiff = np.log10(d_H[:-1]) - np.log10(d_H[1:])
    valdiff = np.exp(logP_dist[1:]) + np.exp(logP_dist[0:-1])
    norm_ = 0.5 * np.sum(valdiff * ddiff, axis=0)

    logP_dist -= np.log(norm_[:, None]).T

    # Calculate the mean and standard deviation of the distribution
    mean = np.sum(dbins[0:-1,None]*np.exp(logP_dist[0:-1])+dbins[1:,None]*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0
    err = np.sqrt(np.sum(dbins[0:-1,None]**2*np.exp(logP_dist[0:-1])+dbins[1:,None]**2*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - mean**2)

    # Calculate the skewness and cap it
    gamma1 = (np.sum(dbins[0:-1,None]**3*np.exp(logP_dist[0:-1])+dbins[1:,None]**3*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - 3.0*mean*err**2 - mean**3)/err**3
    gamma1 = np.where(gamma1 > 0.99, 0.99, gamma1)
    gamma1 = np.where(gamma1 < -0.99, -0.99, gamma1)

    # Calculate the parameters of skew-normal distribution
    delta = np.sign(gamma1)*np.sqrt(np.pi/2.0*1.0/(1.0 + ((4.0 - np.pi)/(2.0*np.abs(gamma1)))**(2.0/3.0)))
    scale = err * np.sqrt(1.0 / (1.0 - 2.0 * delta**2 / np.pi))
    loc = mean - scale * delta * np.sqrt(2.0 / np.pi)
    alpha = delta / (np.sqrt(1.0 - delta**2))

    # # Store the skew-normal values calculated analytically to the dataframe
    # df[f"logdist_mean_{FPlabel.lower()}"] = mean
    # df[f"logdist_std_{FPlabel.lower()}"] = err
    # df[f"logdist_alpha_{FPlabel.lower()}"] = alpha
    # df[f"logdist_loc_{FPlabel.lower()}"] = loc
    # df[f"logdist_scale_{FPlabel.lower()}"] = scale

    # Transpose the PDF and return to linear unit
    y = np.exp(logP_dist.T)

    # Find mean and standard deviation of the distribution using curve_fit
    logdist_mean = []
    logdist_std = []
    chisq = []
    rmse = []
    for i, y_ in enumerate(y):
        popt, pcov = curve_fit(gaus, dbins, y_, p0=[mean[i], err[i]])
        popt[1] = np.absolute(popt[1])
        logdist_mean.append(popt[0])
        logdist_std.append(popt[1])

        # Calculate the chi-squared statistic for the fit
        ypred = norm.pdf(dbins, popt[0], popt[1])
        chisquare = np.sum((y_ - ypred)**2 / ypred, axis=0)
        chisq.append(chisquare)

        # Calculate RMSE statistic for the fit
        ypred = norm.pdf(dbins, popt[0], popt[1])
        rmse_ = np.sqrt((1 / dbins.shape[0]) * np.sum((y_ - ypred)**2, axis=0))
        rmse.append(rmse_)

    df[f'logdist'] = logdist_mean
    df[f'logdist_err'] = logdist_std
    df[f'logdist_chisq'] = chisq
    df[f'logdist_rmse'] = rmse
    
    return df


def main():
    # Load mock data
    df_path = os.path.join(ROOT_PATH, "data/mocks/mock_galaxies/all_combined_mocks_smin_1_fp_fit_method_0.txt")
    df = pd.read_csv(df_path, delim_whitespace=True)
    df["mock_id"] = df["#mockgal_ID"].apply(lambda x: int(x.split("_")[1]))
    df['z'] = df['cz'] / LightSpeed
    df['C_m'] = 1.0

    # Load FP fit
    fp_fit_path = os.path.join(ROOT_PATH, "artifacts/mock_fits/smin_setting_1/fp_fit_method_0/all_combined_fit_with_full_fn.csv")
    fp_fit = pd.read_csv(fp_fit_path)

    for mock_id in df["mock_id"].unique():
        # Filter mock ID
        df_ = df[df["mock_id"] == mock_id].copy()
        fp_fit_ = fp_fit[fp_fit["mock_id"] == mock_id].copy()[fp_fit.columns[1:]].to_numpy()[0]

        df_logdist = fit_logdist(
            df=df_,
            smin=SURVEY_VELDISP_LIMIT[1]['6dFGS'],
            FPparams=fp_fit_,
            use_full_fn=True
        )

        df_logdist.to_csv(os.path.join(f"experiments/experiment_010_mock_logdist/results/mock_{mock_id}.csv"), index=False)


if __name__ == "__main__":
    main()
    