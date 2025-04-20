import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load the data file
# file_path = 'HDPE_torsion_TTS_Layer_1.csv'  
data = pd.read_csv("C:\\Users\\Nishant Bharwani\\Desktop\\Navya\\project\\backend\\services\\HDPE_torsion_TTS_Layer_1.csv")
# Calculate the new shift factors with 30 as the reference temperature
original_shift_factors = {
30:1.00000000000000000000,
40:0.03031005859374914096,
50:0.00142669677734286312,
60:0.00009188652038485410,
70:0.00000605583190829151,
80:0.00000039394944817417,
90:0.00000002444721669548,
100:0.00000000140571412288,
110:0.00000000009213092511,
120:0.00000000000327329275,
}




estimated_shift_factors = {}
for reference_temp in original_shift_factors.keys():
    scale_factor = 1 / original_shift_factors[reference_temp]
    estimated_shift_factors[reference_temp] = {temp: factor * scale_factor for temp, factor in original_shift_factors.items()}

# Convert data to floating point
data['Frequency'] = data['Frequency'].astype(float)
data['Storage Modulus'] = data['Storage Modulus'].astype(float)

# Extract unique temperatures
temperatures = data['Temperature'].unique()

# Define the model function based on the provided equation
def storage_modulus_model(log_omega, a, b, c, d):
    return a * np.tanh(b * (log_omega + c)) + d

# Define bounds for the curve fitting parameters
lower_bounds = [1e-6, -1000, -1000, 1e-6]  ##  Add scrowbar
upper_bounds = [1000, 1000, 1000, 1000]  ##  Add scrowbar

# Creating a new figure for the plot
plt.figure(figsize=(12, 8))

# Define a colormap
n_ref_temps = len(estimated_shift_factors.keys())
colors = cm.rainbow(np.linspace(0, 1, n_ref_temps))

# List to store DataFrames for each reference temperature
all_data_frames = []
# List to store parameters for each reference temperature
abcd_parameters = []
# Dictionary to store the Coefficient of Determination (R^2) for each fitting
r2_scores = {}
# Looping over each reference temperature for plotting and fitting
for ref_temp, color in zip(sorted(estimated_shift_factors.keys()), colors):
    ref_shift_factors = estimated_shift_factors[ref_temp]
    combined_log_freq = []
    combined_storage_modulus = []

    # Plotting each temperature's data
    for temp in sorted(temperatures):
        if temp in ref_shift_factors:
            subset = data[data['Temperature'] == temp].copy()
            subset['Adjusted Frequency (Hz)'] = subset['Frequency'] * ref_shift_factors[temp]
            combined_log_freq.extend(np.log10(subset['Adjusted Frequency (Hz)']))
            combined_storage_modulus.extend(subset['Storage Modulus'])

            plt.semilogx(subset['Adjusted Frequency (Hz)'], subset['Storage Modulus'], 'o', markersize=4, color=color, label=f'Ref Temp {ref_temp}°C' if temp == min(temperatures) else "")

    # Perform curve fitting
    combined_log_freq = np.array(combined_log_freq)
    combined_storage_modulus = np.array(combined_storage_modulus)
    params, _ = curve_fit(storage_modulus_model, 
                combined_log_freq, 
                combined_storage_modulus, 
                bounds=(lower_bounds, upper_bounds), 
                maxfev=9999999
    )
    
    # Extract the estimated parameters
    a, b, c, d = params
    abcd_parameters.append((ref_temp, a, b, c, d))  # Append parameters with reference temperature
    # Plotting the fitted curve
    fitted_storage_modulus = storage_modulus_model(combined_log_freq, *params)
    plt.plot(10**combined_log_freq, fitted_storage_modulus, color='black', linestyle='--', linewidth=1)
    
    r2 = r2_score(combined_storage_modulus, fitted_storage_modulus)
    r2_scores[ref_temp] = r2

    # Create a DataFrame for the current reference temperature
    df = pd.DataFrame({
        'Reference Temperature (°C)': [ref_temp] * len(combined_log_freq),
        'Reduced Frequency (Hz)': 10**combined_log_freq,
        'Storage Modulus (MPa)': combined_storage_modulus,
        'Fitted Storage Modulus (MPa)': fitted_storage_modulus
    })
    all_data_frames.append(df)

# Adding labels, title, grid, and legend
plt.xlabel('Reduced Frequency (Hz)', fontsize=24)
plt.ylabel('Storage Modulus (MPa)', fontsize=24)
#plt.xlim(1e-12, 1e12)
plt.yscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=10)
plt.title('Master Curve for All Reference Temperatures', fontsize=24)

# Show plot
plt.show()

for params in abcd_parameters:
    ref_temp, a, b, c, d = params
    print(f"Ref Temp {ref_temp}°C - A: {a}, B: {b}, C: {c}, D: {d}")
    
# Print the R^2 scores
for ref_temp, r2 in r2_scores.items():
    print(f"Ref Temp {ref_temp}°C: R² = {r2:.9f}")

# #Concatenate all DataFrames and save to a CSV file
# master_curve_data = pd.concat(all_data_frames)
# master_curve_data.to_csv('Rohacell_Master_Curve_and_fitted_Data.csv', index=False)
