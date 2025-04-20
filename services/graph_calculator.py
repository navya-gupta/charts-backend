import numpy as np
from scipy.integrate import simps
from scipy.optimize import curve_fit
from services.graph_functions import get_abcd_parameters
from sklearn.metrics import r2_score
import json
# import pandas as pd

def calculate_relaxation_modulus_vs_time(data, shift_factors, a_upper_bound=500, d_upper_bound=500):
    # Define the model function
    def storage_modulus_model(log_omega, a, b, c, d):
        return a * np.tanh(b * (log_omega + c)) + d

    # Prepare combined data
    combined_log_freq = []
    combined_storage_modulus = []

    for temp in data['Temperature'].unique():
        temp_data = data[data['Temperature'] == temp]
        shifted_log_freq = np.log10(temp_data['Frequency'] * shift_factors[temp])
        combined_log_freq.extend(shifted_log_freq)
        combined_storage_modulus.extend(temp_data['Storage Modulus'])

    combined_log_freq = np.array(combined_log_freq)
    combined_storage_modulus = np.array(combined_storage_modulus)

    # Curve fitting with bounds
    lower_bounds = [1e-6, -10, -10, 1e-6]
    upper_bounds = [a_upper_bound, 10, 10, d_upper_bound]

    params, _ = curve_fit(
        storage_modulus_model,
        combined_log_freq,
        combined_storage_modulus,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000
    )

    a, b, c, d = params

    # Define E'(w)
    def E_prime(w):
        return a * np.tanh(b * ((np.log(w)) + c)) + d

    # Function for E(t)
    def Etime_time_cycle(time, cycle=100):
        N1, N2, N3 = 240, 74, 24
        Etime = np.zeros_like(time)

        def integrand(t, E_prime_w, w):
            return (2 / np.pi) * (E_prime_w / w) * np.sin(w * t)

        for i, t in enumerate(time):
            w1 = np.linspace((1e-6 / t), (cycle * 0.1 * 2 * np.pi / t), int(cycle * 0.1 * N1 + 1))
            w2 = np.linspace((cycle * 0.1 * 2 * np.pi) / t, (cycle * 0.4 * 2 * np.pi) / t, int(cycle * 0.3 * N2 + 1))
            w3 = np.linspace((cycle * 0.4 * 2 * np.pi) / t, (cycle * 2 * np.pi) / t, int(cycle * 0.6 * N3 + 1))
            all_w = np.concatenate([w1, w2[1:], w3[1:]])
            y = integrand(t, E_prime(all_w), all_w)
            Etime[i] = np.trapz(y, all_w)

        return Etime

    # Generate time array and compute E(t)
    time = np.logspace(-10, 10, 500)
    Etime = Etime_time_cycle(time) / 2

    # Prepare data for frontend
    et_data = [{"time": t, "Etime": e} for t, e in zip(time, Etime)]

    return et_data
    


def calculate_sheer_modulus_vs_frequency(data):
    data['Frequency'] = data['Frequency'].astype(float)
    data['Storage Modulus'] = data['Storage Modulus'].astype(float)

    # Extract unique temperatures
    temperatures = data['Temperature'].unique()

    output_data = []
    for temp in temperatures:
        subset = data[data['Temperature'] == temp]
        temp_data = {
            "temperature": f"{temp} °C",
            "data": [
                {"Frequency": row['Frequency'], "StorageModulus": row['Storage Modulus']}
                for _, row in subset.iterrows()
            ]
        }
        output_data.append(temp_data)

    return output_data



def calculate_master_curve_graph_data(data, a_upper_bound=500, d_upper_bound=500):
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
    upper_bounds = [a_upper_bound, 1000, 1000, d_upper_bound]  ##  Add scrowbar



    # List to store DataFrames for each reference temperature
    all_curves = []
    # List to store parameters for each reference temperature
    abcd_parameters = []
    # Dictionary to store the Coefficient of Determination (R^2) for each fitting
    r2_scores = {}
    # Looping over each reference temperature for plotting and fitting
    for ref_temp in sorted(estimated_shift_factors.keys()):
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
        
        r2 = r2_score(combined_storage_modulus, fitted_storage_modulus)
        r2_scores[ref_temp] = r2

        # Create a DataFrame for the current reference temperature
        # df = pd.DataFrame({
        #     'Reference Temperature (°C)': [ref_temp] * len(combined_log_freq),
        #     'Reduced Frequency (Hz)': 10**combined_log_freq,
        #     'Storage Modulus (MPa)': combined_storage_modulus,
        #     'Fitted Storage Modulus (MPa)': fitted_storage_modulus
        # })

        curve_data = {
            "reference_temp": ref_temp,
            "data": [{"frequency": 10**log_f, "storage_modulus": sm, "fitted_storage_modulus": fsm}
                     for log_f, sm, fsm in zip(combined_log_freq, combined_storage_modulus, fitted_storage_modulus)]
        }

        all_curves.append(curve_data)



    print(all_curves)

    # return json.dumps(all_curves)
    return all_curves







def calculate_graph_data(data):

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


    print("abcd_parameters: ", abcd_parameters)

    strain_min = 1e-25
    strain_max = 0.0025
    num_steps = 500
    strain_rates_to_plot = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    results = []

    def E_prime(w, a, b, c, d):
        return a * np.tanh(b * ((np.log(w)) + c)) + d
    
    def Etime_time_cycle(time, cycle, a, b, c, d):
        N1, N2, N3 = 240, 74, 24
        Etime = np.zeros_like(time)

        def integrand(t, E_prime_w, w):
            return (2/np.pi)*(E_prime_w/w)*np.sin(w*t)

        for i, t in enumerate(time):
            w1 = np.linspace((1e-6 / t), (cycle * 0.1 * 2 * np.pi / t), int(cycle * 0.1 * N1 + 1))
            w2 = np.linspace((cycle * 0.1 * 2 * np.pi) / t, (cycle * 0.4 * 2 * np.pi) / t, int(cycle * 0.3 * N2 + 1))
            w3 = np.linspace((cycle * 0.4 * 2 * np.pi) / t, (cycle * 2 * np.pi) / t, int(cycle * 0.6 * N3 + 1))
            all_w = np.concatenate([w1, w2[1:], w3[1:]])
            y = integrand(t, E_prime(all_w, a, b, c, d), all_w)
            Etime[i] = np.trapz(y, all_w)

        return Etime
    
    for params in abcd_parameters:
        ref_temp, a, b, c, d = params
        final_cumulative_integrals = []

        # Calculate the final cumulative integral for each strain rate
        for rate in strain_rates_to_plot:
            # Calculate the time range corresponding to the strain range
            time_min_rate = strain_min / rate
            time_max_rate = strain_max / rate

            # Generate a linear time array within the specified range
            time_range_rate = np.linspace(time_min_rate, time_max_rate, num_steps)

            # Calculate E(t) for the linear time array within the integration range
            E_t_time_range_rate = Etime_time_cycle(time_range_rate, 500, a, b, c, d)

            # Multiply E(t) by the strain rate at each time point
            Stress_history_rate = E_t_time_range_rate * rate

            # Create a cumulative integral array for plotting
            cumulative_integral_stress_history_rate = np.array([simps(Stress_history_rate[:i+1], time_range_rate[:i+1]) for i in range(len(time_range_rate))])
            cumulative_integral_stress_history_rate = cumulative_integral_stress_history_rate[-1]/strain_max

            # Append the final value of the cumulative integral to the list
            final_cumulative_integrals.append(cumulative_integral_stress_history_rate)

        # Append the results for the current reference temperature to the results list
        results.append([ref_temp] + final_cumulative_integrals)

    print(results)
    return results
    
# # Create a DataFrame from the results list
# df = pd.DataFrame(results, columns=['Ref Temp (°C)'] + [f'Strain Rate {rate} (1/s)' for rate in strain_rates_to_plot])

# # Plot 1: x is strain rate, y is modulus for every temperature
# plt.figure(figsize=(12, 6))
# for i, row in df.iterrows():
#     plt.plot(strain_rates_to_plot, row[1:], marker='o', label=f"{row['Ref Temp (°C)']}°C")

# plt.xscale('log')
# plt.xlabel('Strain Rate (1/s)', fontsize=20)
# plt.ylabel('Modulus', fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.title('Modulus vs Strain Rate for Different Temperatures')
# plt.legend(fontsize=15)
# plt.show()

# # Plot 2: x is temperature, y is modulus at every strain rate
# plt.figure(figsize=(12, 6))
# for j, rate in enumerate(strain_rates_to_plot):
#     plt.plot(df['Ref Temp (°C)'], df.iloc[:, j+1], marker='s', label=f"Strain Rate {rate} (1/s)")

# plt.xlabel('Temperature (°C)', fontsize=20)
# plt.ylabel('Modulus', fontsize=20)
# plt.title('Modulus vs Temperature for Different Strain Rates')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=15)
# plt.show()

# # # Save the DataFrame to a CSV file
# df.to_csv('Software_G_vs_strain_rate.csv', index=False)  ## Creat a CSV file for User

# # # Print a message to indicate that the CSV file has been saved
# print('Results saved to final_cumulative_integral_results.csv')