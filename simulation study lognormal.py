# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 10:43:39 2025

@author: Bhadrar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 13:18:43 2025

@author: Bhadrar
"""

import numpy as np
from scipy.stats import lognorm
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar


import numpy as np
from scipy.stats import lognorm
import math

def simulate_marked_nhpp(T, a, b, m0, m1, cv0, cv1, seed=None):
    """
    Simulate a marked NHPP where marks follow a lognormal distribution with time-varying mean and CV.
    """
    if seed is not None:
        np.random.seed(seed)

    # Define λ_max for thinning
    t_vals = np.linspace(0, T, 1000)
    lambda_vals = np.exp(a + b * t_vals)
    lambda_max = np.max(lambda_vals)

    # Generate homogeneous Poisson process
    N_star = np.random.poisson(lambda_max * T)
    t_star = np.sort(np.random.uniform(0, T, N_star))

    events = []
    magnitudes = []

    for t in t_star:
        if np.random.uniform() <= np.exp(a + b * t) / lambda_max:
            events.append(t)
            
            # Time-varying mean and CV
            mean_t = m0 + m1 * t
            cv_t = cv0 + cv1 * t

            # Convert to lognormal parameters
            sigma_ln = np.sqrt(np.log(1 + cv_t**2))
            mu_ln = np.log(mean_t) - 0.5 * sigma_ln**2
            
            # Sample from lognormal
            magnitude = lognorm.rvs(s=sigma_ln, scale=np.exp(mu_ln))
            magnitudes.append(magnitude)

    return events, magnitudes

def lognormal_cdf(x, mean, cv):
    """CDF of lognormal with given mean and CV."""
    sigma_ln = np.sqrt(np.log(1 + cv**2))
    mu_ln = np.log(mean) - 0.5 * sigma_ln**2
    return lognorm.cdf(x, s=sigma_ln, scale=np.exp(mu_ln))

def nhpp_cdf_grid(year, a, b, m0, m1, cv0, cv1, x_grid):
    """NHPP maximum CDF over one year using lognormal marks."""
    F_vals = []

    for x in x_grid:
        def integrand(t):
            lam = np.exp(a + b * t)
            mean = m0 + m1 * t
            cv = cv0 + cv1 * t
            return lam * (1 - lognormal_cdf(x, mean, cv))

        # Approximate 1-year integral as point evaluation at `year`
        val = integrand(year)
        F_vals.append(np.exp(-val))

    return np.array(F_vals)

def nhpp_yearwise_percentiles_interp(T, a, b, m0, m1, cv0, cv1, percentiles=[50, 95, 98]):
    """Compute year-wise percentiles from NHPP using lognormal marks."""
    x_grid = np.linspace(0.0, 5, 25)
    results = {p: [] for p in percentiles}

    for year in range(T):
        F_vals = nhpp_cdf_grid(year, a, b, m0, m1, cv0, cv1, x_grid)
        for p in percentiles:
            percentile_value = np.interp(p / 100.0, F_vals, x_grid, left=np.nan, right=np.nan)
            results[p].append(percentile_value)

    return results



        

def get_annual_maxima(event_times, event_magnitudes,total_years):
    """
    Computes annual maximum magnitudes from event time and magnitude vectors.

    Parameters:
    - event_times: np.array of event times in years (e.g., 0.0 to 49.99 for 50 years)
    - event_magnitudes: np.array of associated magnitudes
    

    Returns:
    - annual_maxima: np.array of length `total_years` with max value per year (NaN if no event)
    """
    # Ensure input is in NumPy array form
    event_times = np.asarray(event_times)
    event_magnitudes = np.asarray(event_magnitudes)
    
    annual_maxima = np.full(total_years, np.nan)
    for year in range(total_years):
        mask = (event_times > year) & (event_times < year + 1)
        if np.any(mask):
            annual_maxima[year] = np.max(event_magnitudes[mask])
    return np.array(annual_maxima)

import numpy as np
from scipy.stats import genextreme as gev
from scipy.optimize import minimize

def fit_nonstationary_gev(annual_maxima_matrix, percentiles=[95, 98]):
    n_sim, n_years = annual_maxima_matrix.shape
    years = np.arange(n_years)

    # Flatten the matrix and create the time vector
    all_data = annual_maxima_matrix.flatten()
    time = np.tile(np.arange(annual_maxima_matrix.shape[1]), annual_maxima_matrix.shape[0])
    
    # Remove NaNs from both all_data and corresponding time entries
    valid_mask = ~np.isnan(all_data)
    all_data = all_data[valid_mask]
    time = time[valid_mask]
    
    # Extract and clean first year's data (for fixed mu0, sigma0, xi estimation)
    year0_data = annual_maxima_matrix[:, 0]
    year0_data = year0_data[~np.isnan(year0_data)]

    # Fit stationary GEV to first year's data
    c0, loc0, scale0 = gev.fit(year0_data)  # c = -xi in scipy
    xi = -c0
    mu0 = loc0
    sigma0 = scale0

    print(f"Fixed from year 0: mu0 = {mu0:.3f}, sigma0 = {sigma0:.3f}, xi = {xi:.3f}")

    # Define negative log-likelihood with time-varying mu and sigma
    def neg_log_likelihood(params):
        mu1, sigma1 = params
        mu_t = mu0 * np.exp(mu1 * time)
        sigma_t = sigma0 * np.exp(sigma1 * time)

        if np.any(sigma_t <= 0):
            print('mu1=',mu1,'sigma1=',sigma1,'error: sigmat <0')
            return 1e10

        z = (all_data - mu_t) / sigma_t
        t_check = 1 + xi * z
        if np.any(t_check <= 0) or np.any(np.isnan(t_check)):
            print('mu1=',mu1,'sigma1=',sigma1,'t_check=',min(t_check),'error: np.any(t_check <= 0) or np.any(np.isnan(t_check))')
            return 1e10

        try:
            logpdf = gev.logpdf(all_data, c=-xi, loc=mu_t, scale=sigma_t)
            if np.any(np.isnan(logpdf)) or np.any(np.isinf(logpdf)):
                print('mu1=',mu1,'sigma1=',sigma1,'error: np.any(np.isnan(logpdf)) or np.any(np.isinf(logpdf))')
                return 1e10
            print('mu1=',mu1,'sigma1=',sigma1,'-sum(logpdf)',-np.sum(logpdf))
            return -np.sum(logpdf)
        except Exception as e:
            print("Numerical error:", e)
            return 1e10


    # Initial guesses
    mu1_init = 0.0092
    sigma1_init = 0.0051
    init_params = [mu1_init, sigma1_init]

    bounds = [(-1, 1), (-1, 1)]

    result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds)#,options={'ftol': 1e-10, 'gtol': 1e-8, 'disp': True, 'maxiter': 1000})

    if not result.success:
        raise RuntimeError("GEV fit did not converge: " + result.message)

    mu1, sigma1 = result.x
    print(f"Fitted mu1 = {mu1:.4f}, sigma1 = {sigma1:.4f}")

    # Compute percentiles
    mu_t = mu0 * np.exp(mu1 * years)
    sigma_t = sigma0 * np.exp(sigma1 * years)
    percentiles_dict = {}

    for p in percentiles:
        q = gev.ppf(p / 100, c=-xi, loc=mu_t, scale=sigma_t)
        percentiles_dict[p] = q

    return percentiles_dict



# Parameters of the marked NHPP to be simulated
N_simulations = 1000
T = 100              # time horizon in years
a, b = 0.5, 0.02     # log-linear rate parameters
m0, m1 = 1.0, 0.30/T   # time-varying Weibull mean
cv0, cv1 = 0.5, 0.1/T # time-varying CV

percentiles = [95, 98]

#Simulating the event times and magnitudes of the marked-NHPP process
all_maxima = np.zeros((N_simulations, T))
for ii in range(N_simulations):
    events, magnitudes = simulate_marked_nhpp(T, a, b, m0, m1, cv0, cv1, seed=None)
    ann_max = get_annual_maxima(events,magnitudes,T) 
    all_maxima[ii, :] = ann_max
    

# Compute statistics from empirical annual maximas
median_max = np.nanmedian(all_maxima, axis=0)
p95_max = np.nanpercentile(all_maxima, 95, axis=0)
p98_max = np.nanpercentile(all_maxima, 98, axis=0)
years = np.arange(T)

# Computing the percentiles by fitting a ns-gev distribution
percentiles_dict = fit_nonstationary_gev(all_maxima)

#Obtaining the analytical nhpp percentiles
yearwise_results = nhpp_yearwise_percentiles_interp(T, a, b, m0, m1, cv0, cv1, percentiles)

#%%
import matplotlib.pyplot as plt
import numpy as np

# Use clean and consistent style
plt.style.use("seaborn-v0_8-whitegrid")

# Create figure and axis
fig, ax = plt.subplots(figsize=(6,4))  # 8x6 inches for publication

# Scatter plots for empirical percentiles
ax.scatter(years, p95_max, label='95th Percentile (Empirical)', color='k', s=10, marker='o')
ax.scatter(years, p98_max, label='98th Percentile (Empirical)', color='k', s=10, marker='^')

# NHPP model percentile curves
for p in percentiles:
    ax.plot(years, yearwise_results[p], label=f'{p}th Percentile (NHPP)', linestyle='-', linewidth=2)

# GEV-fitted percentile curves
for p, curve in percentiles_dict.items():
    ax.plot(years, curve, label=f"{p}th Percentile (GEV)", linewidth=2,linestyle = '--')

# Axis labels
ax.set_xlabel("Year", fontsize=14, fontweight='bold')
ax.set_ylabel("Annual Maximum (unit)", fontsize=14, fontweight='bold')

# Ticks
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.minorticks_on()

# Grid
ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# Legend
ax.legend(fontsize=12, loc="upper left", frameon=False)

# Tight layout
plt.tight_layout()

# Save to PDF
output_filename = "simulation study lognormal.pdf"
plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches="tight")

# Show plot (can be skipped if only saving)
plt.show()

print(f"Plot saved as: {output_filename}")


  
    
