import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.integrate import simps
import pandas as pd


def E_prime(w, a, b, c, d):
    return a * np.tanh(b * ((np.log(w)) + c)) + d


def get_shift_factors(data):
    temperatures = sorted(data['Temperature'].unique())
    temp_data = {temp: data[data['Temperature'] == temp] for temp in temperatures}

    def shift_difference_final(shift_factor, reference_data, target_data, lower_Hz=0.1):
        shifted_frequencies = target_data['Frequency'] * shift_factor
        min_freq = max(reference_data['Frequency'].min(), shifted_frequencies.min()) * lower_Hz
        max_freq = min(reference_data['Frequency'].max(), shifted_frequencies.max())

    #     print(min_freq, max_freq)
        if min_freq >= max_freq:
            return np.inf # Return a large value if there is no overlap

        common_frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), 10)
        ref_modulus_interp = np.interp(common_frequencies, reference_data['Frequency'], reference_data['Storage Modulus'])
        target_modulus_interp = np.interp(common_frequencies, shifted_frequencies, target_data['Storage Modulus'])

        if np.isnan(ref_modulus_interp).any() or np.isnan(target_modulus_interp).any():
            return np.inf
    #     print("KK for calc shift fact 0")
        difference = np.sum((ref_modulus_interp - target_modulus_interp)**2)
        return difference
    
    # Initialize shift_factors with the lowest temperature set to 1
    shift_factors = {temperatures[0]: 1}  # Set the lowest temperature's shift factor to 1
    ref_temp = temperatures[0]
    ref_data = temp_data[ref_temp]

    for i, temp in enumerate(temperatures[1:], start=1):
        target_data = temp_data[temp]
        result = minimize(shift_difference_final, 1, args=(ref_data, target_data), method='Nelder-Mead')
        shift_factors[temp] = result.x[0]
        ref_data = target_data.copy()
        ref_data['Frequency'] = ref_data['Frequency'] * result.x[0]

    # print("KK for calc shift fact 3")
    return shift_factors



def get_abcd_parameters(file_path: str):
    # data = pd.read_csv("C:\\Users\\Nishant Bharwani\\Desktop\\Navya\\project\\backend\\services\\HDPE_torsion_TTS_Layer_1.csv")
    data = pd.read_csv(file_path)
    print(data)
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
    
    data['Frequency'] = data['Frequency'].astype(float)
    data['Storage Modulus'] = data['Storage Modulus'].astype(float)

    # Extract unique temperatures
    temperatures = data['Temperature'].unique()
    # Define the model function based on the provided equation
    def storage_modulus_model(log_omega, a, b, c, d):
        return a * np.tanh(b * (log_omega + c)) + d
    
    lower_bounds = [1e-6, -1000, -1000, 1e-6]  ##  Add scrowbar
    upper_bounds = [1000, 1000, 1000, 1000]  ##  Add scrowbar

    n_ref_temps = len(estimated_shift_factors.keys())
    abcd_parameters = []

    for ref_temp in sorted(estimated_shift_factors.keys()):
        ref_shift_factors = estimated_shift_factors[ref_temp]
        combined_log_freq = []
        combined_storage_modulus = []

        for temp in sorted(temperatures):
            if temp in ref_shift_factors:
                subset = data[data['Temperature'] == temp].copy()
                subset['Adjusted Frequency (Hz)'] = subset['Frequency'] * ref_shift_factors[temp]
                combined_log_freq.extend(np.log10(subset['Adjusted Frequency (Hz)']))
                combined_storage_modulus.extend(subset['Storage Modulus'])

        combined_log_freq = np.array(combined_log_freq)
        combined_storage_modulus = np.array(combined_storage_modulus)
        params, _ = curve_fit(storage_modulus_model, combined_log_freq, combined_storage_modulus, bounds=(lower_bounds, upper_bounds),maxfev=9999999)
        a, b, c, d = params

        abcd_parameters.append((ref_temp, a, b, c, d))

    return abcd_parameters