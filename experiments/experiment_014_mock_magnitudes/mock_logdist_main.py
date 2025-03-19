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
from src.utils.functions import create_parent_folder

np.random.seed(42)

def fit_mock(
        df: pd.DataFrame,
        smin: float,
        mag_lim: str,
        use_full_fn: bool = True,
        reject_outliers: bool = False,
        pvals_cut: float = PVALS_CUT
        ):
    """_summary_

    Args:
        mock_filename (str): _description_
        survey (str): _description_
        fp_fit_method (int): _description_
        smin_setting (int): _description_
        id_start (int): _description_
        id_end (int): _description_
        output_filepath (str): _description_
    """

    # Calculate redshift
    df['z'] = df['cz'] / LIGHTSPEED

    # Calculate predicted true distance and FN integral limits
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    d_H = sp.interpolate.splev(df['z'].to_numpy(), dist_spline, der=0)
    df['lmin'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - float(mag_lim)) / 5.0
    df['lmax'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_LOW) / 5.0

    # Assume 0 peculiar velocities
    df['logdist_pred'] = 0.0
    df['r_true'] = df['r'] - df['logdist_pred']

    if not use_full_fn:
        Sn = df["Sprob"].to_numpy()
    else:
        Sn = 1.0

    df['Sn'] = Sn
    df['C_m'] = 1.0

    # Range of FP parameters
    param_boundaries = [(1.4, 2.0), (-1.1, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.1, 3.5), (0.0, 0.06), (0.25, 0.45), (0.14, 0.25)]

    avals, bvals = param_boundaries[0], param_boundaries[1]
    rvals, svals, ivals = param_boundaries[2], param_boundaries[3], param_boundaries[4]
    s1vals, s2vals, s3vals = param_boundaries[5], param_boundaries[6], param_boundaries[7]

    # Fitting the FP iteratively
    data_fit = df
    badcount = len(df)
    is_converged = False
    i = 1
    while not is_converged:

        Snfit = data_fit["Sn"].to_numpy()

        # Fit the FP parameters
        FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
            args=(0.0, data_fit["z"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["dr"].to_numpy(), data_fit["ds"].to_numpy(), data_fit["di"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), True, False, use_full_fn), maxiter=10000, tol=1.0e-6, workers=-1)

        # Break from the loop if not iterative
        if reject_outliers == False:
            break

        # Calculate the chi-squared 
        chi_squared = Sn * FP_func(FPparams.x, 0.0, df["z"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["dr"].to_numpy(), df["ds"].to_numpy(), df["di"].to_numpy(), Sn, smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, chi_squared_only=True, use_full_fn=use_full_fn)[0]
        
        # Calculate the p-value (x,dof)
        pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
        
        # Reject galaxies with p-values < pvals_cut (probabilities of being part of the sample lower than some threshold)
        data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
        
        # Count the number of rejected galaxies
        badcountnew = len(np.where(pvals < pvals_cut)[0])
        
        # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
        is_converged = True if badcount == badcountnew else False
        
        # Set the new count of rejected galaxies
        badcount = badcountnew
        i += 1

    return FPparams.x


def fit_logdist(
        df: pd.DataFrame,
        smin: float,
        FPparams: np.ndarray,
        use_full_fn: bool = True,
        mag_high: str = str(MAG_HIGH),
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
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - float(mag_high)) / 5.0
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
    input_path = os.path.join(ROOT_PATH, "experiments/experiment_014_mock_magnitudes/mock_full.txt")
    df_mock = pd.read_csv(input_path, delim_whitespace=True)
    df_mock["z"] = df_mock["cz"] / LightSpeed
    df_mock["C_m"] = 1.0

    # Extract mock number
    df_mock["mock_no"] = df_mock["#mockgal_ID"].apply(lambda x: int(x.split("_")[1]))
    mock_ids = df_mock["mock_no"].unique().tolist()
    
    # Select subsample based on magnitude upper limit
    mag_lims = ["14.0", "13.5", "13.0", "12.5", "12.0"]

    # For each mock sample...
    for mock_id in sorted(mock_ids):
        print("Processing mock ", mock_id)
        df_ = df_mock.copy()
        df_ = df_[df_["mock_no"] == mock_id]

        fp_params_list = []
        # Apply each magnitude limit...
        for mag_lim in mag_lims:
            print("Processing sample with limiting magnitude of ", mag_lim)

            df = df_.copy()

            # Apply magnitude limit
            df = df[(df["mag_j"] - df["extinction_j"]) <= float(mag_lim)]

            print("Fitting FP Parameters...")
            fp_params = fit_mock(
                df=df,
                smin=2.0,
                mag_lim=mag_lim,
                use_full_fn=False
            )

            # Create dictionary of FP params
            fp_params_dict = {
                "mag_lim": mag_lim,
                "a": fp_params[0],
                "b": fp_params[1],
                "rmean": fp_params[2],
                "smean": fp_params[3],
                "imean": fp_params[4],
                "sigma1": fp_params[5],
                "sigma2": fp_params[6],
                "sigma3": fp_params[7],
            }
            fp_params_list.append(fp_params_dict)

            # # Append the results to the output file
            # with open(f"/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_014_mock_magnitudes/mock_{mock_id}/fp_params_{mag_lim}.txt", "wt") as myfile:
            #     text = ','.join([str(x) for x in FPparams.x]) + '\n'
            #     myfile.write(text)

            print("Fitting log-distance ratios...")
            df_logdist = fit_logdist(
                df=df,
                smin=2.0,
                FPparams=fp_params,
                use_full_fn=True,
                mag_high=mag_lim
            )

            logdist_filepath = os.path.join(ROOT_PATH, f"experiments/experiment_014_mock_magnitudes/logdists/mock_{mock_id}/mock_{mag_lim}.csv")
            create_parent_folder(logdist_filepath)
            df_logdist.to_csv(logdist_filepath, index=False)

        # Save FP parameters for a single mock
        pd.DataFrame(fp_params_list).to_csv(f"/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_014_mock_magnitudes/fp_fits/mock_{mock_id}.csv", index=False)


if __name__ == "__main__":
    main()