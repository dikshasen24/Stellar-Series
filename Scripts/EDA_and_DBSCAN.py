#!/usr/bin/env python
# coding: utf-8

from sklearn.cluster import DBSCAN
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf  # Import the acf function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import interp1d

from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression


# Load data for star
star1 = pd.read_csv('0131799991.csv')

# Raw flux vs time for star 1
plt.plot(star1['time'], star1['pdcsap_flux'], label="Star 1", color="blue")
plt.title("Star: TIC 0131799991")
plt.xlabel("Time")
plt.ylabel("PDCSAP Flux")

plt.show()


star1.head()


star1.shape


# Step 1: Calculate time_diff and identify the time gap
star1['time_diff'] = star1['time'].diff()
largest_diff_index = star1['time_diff'].idxmax()
largest_diff_start = star1.iloc[largest_diff_index - 1]['time']
largest_diff_end = star1.iloc[largest_diff_index]['time']
median_interval = star1['time_diff'].median()

# Step 2: Split data into part1 and part3
part1 = star1[star1['time'] <= largest_diff_start].copy()
part3 = star1[star1['time'] > largest_diff_end].copy()

# Remove NaN values from part1 for STL decomposition
part1 = part1.dropna(subset=['pdcsap_flux'])


# Print the timeframe for which values are missing

# Start
print(largest_diff_start)

# End
print(largest_diff_end)


# Compute autocorrelation values for the first 100 lags
acf_values_500 = acf(part1['pdcsap_flux'], nlags=500, fft=True)

conf_bound = 1.96 / np.sqrt(len(part1))

# Plot autocorrelation for the first 100 lags
plt.figure(figsize=(10, 5))
plt.stem(range(len(acf_values_500)), acf_values_500, use_line_collection=True)
#plt.axhline(y=conf_bound, color='blue', linestyle='dashed', label="95% Conf. Interval")
#plt.axhline(y=-conf_bound, color='blue', linestyle='dashed')
plt.title('Autocorrelation Function Plot (First 500 Lags)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
# plt.legend()
plt.grid()
#plt.savefig("star1_acf.png", dpi=300)
plt.show()


# Compute autocorrelation values for the first 100 lags
pacf_values_500 = pacf(part1['pdcsap_flux'].dropna(), nlags=50)

conf_bound = 1.96 / np.sqrt(len(part1))

# Plot autocorrelation for the first 100 lags
plt.figure(figsize=(10, 5))
plt.stem(
    range(
        len(pacf_values_500)),
    pacf_values_500,
    use_line_collection=True)
plt.axhline(
    y=conf_bound,
    color='blue',
    linestyle='dashed',
    label="95% Conf. Interval")
plt.axhline(y=-conf_bound, color='blue', linestyle='dashed')
plt.title('Partial Autocorrelation Analysis (First 500 Lags)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.grid()
plt.legend()
plt.show()


# Step 3: Model trend and seasonality of part1 using STL decomposition
stl = STL(part1['pdcsap_flux'], period=150)  # 150 chosen based on acf
result = stl.fit()
part1['trend'] = result.trend
part1['seasonal'] = result.seasonal


# Generate new time steps at median intervals
part2_times = np.arange(
    largest_diff_start +
    median_interval,
    largest_diff_end,
    median_interval)

# Create part2 DataFrame to cover the missing times
part2 = pd.DataFrame({'time': part2_times})


# Find the offset to shift part2 to the correct time range
offset = part1.index.max() + 1  # Start part2 after the last index of part1

# Reindex part2 so that it follows part1
part2.index = np.arange(offset, offset + len(part2))

print("Adjusted part2 index range:", part2.index.min(), "-", part2.index.max())


# Extract known x-values and corresponding trend from part1
known_x = part1.index
known_trend = part1['trend'].values

# Fit a linear model to extrapolate the trend
linear_model = np.polyfit(known_x, known_trend, 1)  # Linear regression
predict_trend = np.poly1d(linear_model)  # Create function for prediction

# Predict trend for part2
part2['trend'] = predict_trend(part2.index)


# Extract the last full seasonal cycle from part1
seasonality_period = 150  # Same as STL period
last_seasonal_values = part1['seasonal'].values[-seasonality_period:]

# Extend seasonality for part2 by repeating the pattern
seasonality_predictions = np.tile(last_seasonal_values, len(
    part2) // seasonality_period + 1)[:len(part2)]

# Assign to part2
part2['seasonal'] = seasonality_predictions


# Compute the imputed response for part2
part2['pdcsap_flux'] = part2['trend'] + part2['seasonal']


# Adjust part3 index to continue from part2
part3.index = np.arange(
    part2.index.max() + 1,
    part2.index.max() + 1 + len(part3))

print("Final index ranges:")
print(f"part1: {part1.index.min()} - {part1.index.max()}")
print(f"part2: {part2.index.min()} - {part2.index.max()}")
print(f"part3: {part3.index.min()} - {part3.index.max()}")


plt.figure(figsize=(12, 6))

# Plot part1 (original data)
plt.plot(
    part1.index,
    part1['pdcsap_flux'],
    label="Original Flux",
    color='blue',
    alpha=0.5)

# Plot imputed part2 (predicted response)
plt.plot(part2.index, part2['pdcsap_flux'], label="Imputed Flux", color='red')

# Plot part3 (original data)
plt.plot(part3.index, part3['pdcsap_flux'], color='blue', alpha=0.5)

# Formatting
plt.legend()
plt.xlabel("Time Index")
plt.ylabel("PDCSAP Flux")
plt.title("Star Data with Imputed Values")
#plt.savefig("star1_imputed.png", dpi=300)
plt.show()


# Combine all parts while keeping the correct index
full_df1 = pd.concat(
    [part1[['pdcsap_flux']], part2[['pdcsap_flux']], part3[['pdcsap_flux']]])

# Sort by index to maintain order
full_df1 = full_df1.sort_index()

print(full_df1.head())  # Preview the first few rows
print(full_df1.tail())  # Preview the last few rows


# Make copy of the data and work on copy to avoid errors
star1_data = full_df1.copy()
star1_data = star1_data.reset_index()


# Investigating rolling mean window
for ws in [10, 50, 100, 200]:
    star1_data[f'flux_roll_mean_{ws}'] = star1_data['pdcsap_flux'].rolling(
        window=ws).mean()

plt.figure(figsize=(12, 6))
plt.plot(
    star1_data['index'],
    star1_data['pdcsap_flux'],
    label="Original Flux",
    alpha=0.3,
    linewidth=1)
plt.plot(
    star1_data['index'],
    star1_data['flux_roll_mean_10'],
    label="Rolling Mean (10)",
    linestyle="--")
plt.plot(
    star1_data['index'],
    star1_data['flux_roll_mean_50'],
    label="Rolling Mean (50)",
    linestyle="--")
plt.plot(
    star1_data['index'],
    star1_data['flux_roll_mean_100'],
    label="Rolling Mean (100)",
    linestyle="--")
plt.plot(
    star1_data['index'],
    star1_data['flux_roll_mean_200'],
    label="Rolling Mean (200)",
    linestyle="--")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Effect of Different Window Sizes on Rolling Mean")
plt.show()


# Plot upto t=2000 for better visualization

# Define the time range to plot (up to time = 2000)
time_limit = 2000

# Filter the data up to the specified time limit
filtered_data = star1_data[star1_data['index'] <= time_limit]

plt.figure(figsize=(12, 6))

# Plot the original flux values
plt.plot(
    filtered_data['index'],
    filtered_data['pdcsap_flux'],
    label="Original Flux",
    color='blue',
    alpha=0.1,
    linewidth=1)

# Plot the first few lags
plt.plot(
    filtered_data['index'],
    filtered_data['flux_roll_mean_10'],
    label="Lag 10",
    linestyle='dashed',
    color='red')
plt.plot(
    filtered_data['index'],
    filtered_data['flux_roll_mean_50'],
    label="Lag 50",
    linestyle='dashed',
    color='green')
plt.plot(
    filtered_data['index'],
    filtered_data['flux_roll_mean_100'],
    label="Lag 100",
    linestyle='dashed',
    color='darkorange')
plt.plot(
    filtered_data['index'],
    filtered_data['flux_roll_mean_200'],
    label="Lag 200",
    linestyle='dashed',
    color='black')

# Formatting
plt.legend()
plt.xlabel("Time Index")
plt.ylabel("PDCSAP Flux")
plt.title("Effect of Different Window Sizes on Rolling Mean (Upto Time Index 2000)")
#plt.savefig("star1_rollmean.png", dpi=300)
plt.show()


# Extract additional features from brightness data using chosen rolling window

def preprocess_data(data, window_size=100):
    # Compute flux differences and ratios
    data['flux_diff'] = data['pdcsap_flux'].diff()
    data['flux_ratio'] = data['pdcsap_flux'] / data['pdcsap_flux'].shift(1)

    # Rolling statistics
    data['flux_roll_mean'] = data['pdcsap_flux'].rolling(
        window=window_size, min_periods=1).mean()
    data['flux_roll_std'] = data['pdcsap_flux'].rolling(
        window=window_size, min_periods=1).std()

    return data


data_star1 = preprocess_data(star1_data)
data_star1 = data_star1.iloc[1:].reset_index(
    drop=True)  # drop 1st row with nan
data_star1.head()


# isolate the features from the data

features = data_star1[['flux_diff', 'flux_roll_mean',
                       'flux_roll_std', 'flux_ratio']].values


# Check for NaN or infinity values in the features
print(np.isnan(features).any())  # True if any NaN values are present
print(np.isinf(features).any())  # True if any infinite values are present


# DBSCAN Algorithm


# Define different combinations of (eps, min_samples)
param_combinations = [(0.5, 15), (2, 50), (4, 50)]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)

for i, (eps, min_samples) in enumerate(param_combinations):
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)

    # Add cluster labels back to the original data
    data_star1['cluster'] = clusters

    # Mark anomalies (noise points)
    data_star1['is_flare'] = (
        (data_star1['cluster'] == -
         1) & (
            data_star1['pdcsap_flux'] > data_star1['pdcsap_flux'].median())).astype(int)

    # Filter only flare points
    flare_points = data_star1[data_star1['is_flare'] == 1]

    # Plot results
    ax = axes[i]
    ax.plot(
        data_star1['index'],
        data_star1['pdcsap_flux'],
        label='Flux',
        color='blue')
    ax.scatter(
        flare_points['index'],
        flare_points['pdcsap_flux'],
        color='red',
        label='Detected Flare Points')

    ax.legend()
    ax.set_title(f'Detected Flares (eps={eps}, min_samples={min_samples})')
    ax.set_ylabel('PDCSAP Flux')

# Set x-label only for the last subplot
axes[-1].set_xlabel('Time Index')

plt.tight_layout()
#plt.savefig("star1_flares.png", dpi=300)
plt.show()
