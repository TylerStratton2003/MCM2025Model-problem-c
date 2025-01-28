import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkForAnalysis import predict_medals, build_model, preprocess_data, \
    load_and_merge_data

# Load input files
medal_counts = pd.read_csv("summerOly_medal_counts.csv")
hosts = pd.read_csv("summerOly_hosts.csv")

# Load predictions (output file)
predictions = pd.read_csv('results_file.csv', encoding='latin1')

# Baseline predictions
baseline_gold = predictions['Gold']
baseline_silver = predictions['Silver']
baseline_bronze = predictions['Bronze']


# Function to vary a feature
def vary_feature(input_file, feature_name, variation_range):
    modified_inputs = []
    for delta in variation_range:
        # Create a copy of the input file
        input_copy = input_file.copy()
        # Apply the variation and clip to avoid negative values
        input_copy[feature_name] = (input_copy[feature_name] + delta).clip(lower=0)
        modified_inputs.append(input_copy)
    return modified_inputs


# Vary the 'Gold' column in medal_counts
varied_medal_counts = vary_feature(medal_counts, 'Gold', variation_range=[-5, 0, 5])

# Load and preprocess data
df_merged = load_and_merge_data("summerOly_medal_counts.csv", "summerOly_hosts.csv")
X, y, encoder, scaler = preprocess_data(df_merged)

# Build and load the trained model
model = build_model(input_dim=X.shape[1])

# Assuming the model is already trained and you have a saved version
# model.load_weights('model_weights.h5')  # Uncomment if you have saved model weights

# Placeholder to store results
sensitivity_results = {
    'Gold_variation': [],
    'Delta_Gold': [],
    'Delta_Silver': [],
    'Delta_Bronze': []
}

# Assume Los Angeles, United States is the host and year is 2028
host = "Los Angeles, United States"
year = 2028

# List of countries for prediction (shortened for simplicity)
countries = [
    "Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola",
    "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegovina",
    "Botswana", "Brazil", "British Virgin Islands", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde",
    "Cayman Islands", "Central African Republic", "Chad", "Chile", "China",
    "Chinese Taipei", "Colombia", "Comoros", "Congo", "Cook Islands", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czech Republic", "Democratic Republic of the Congo",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador",
    "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini",
    "Ethiopia", "Federated States of Micronesia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Great Britain", "Greece", "Grenada",
    "Guam", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras",
    "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
    "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto Rico", "Qatar", "Refugee Olympic Team", "Romania",
    "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "San Marino", "São Tomé and Príncipe", "Saudi Arabia", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan",
    "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Venezuela", "Vietnam", "Virgin Islands", "Yemen", "Zambia", "Zimbabwe"
]

# Conduct sensitivity analysis
for modified_medal_counts in varied_medal_counts:
    # Generate predictions using the modified input
    predictions = predict_medals(model, encoder, scaler, year, host, countries)

    # Extract predictions for each medal type
    predicted_gold = [pred['Gold'] for pred in predictions]
    predicted_silver = [pred['Silver'] for pred in predictions]
    predicted_bronze = [pred['Bronze'] for pred in predictions]

    # Calculate the average change in predictions
    avg_delta_gold = np.mean(np.array(predicted_gold) - baseline_gold)
    avg_delta_silver = np.mean(np.array(predicted_silver) - baseline_silver)
    avg_delta_bronze = np.mean(np.array(predicted_bronze) - baseline_bronze)

    # Log the results
    delta_value = modified_medal_counts['Gold'].iloc[0] - medal_counts['Gold'].iloc[0]
    sensitivity_results['Gold_variation'].append(delta_value)
    sensitivity_results['Delta_Gold'].append(avg_delta_gold)
    sensitivity_results['Delta_Silver'].append(avg_delta_silver)
    sensitivity_results['Delta_Bronze'].append(avg_delta_bronze)

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(sensitivity_results)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(results_df['Gold_variation'], results_df['Delta_Gold'], label='Gold Prediction Change', marker='o')
plt.plot(results_df['Gold_variation'], results_df['Delta_Silver'], label='Silver Prediction Change', marker='o')
plt.plot(results_df['Gold_variation'], results_df['Delta_Bronze'], label='Bronze Prediction Change', marker='o')
plt.xlabel('Variation in Gold Medals')
plt.ylabel('Change in Prediction')
plt.title('Sensitivity Analysis for Gold Medals')
plt.legend()
plt.grid(True)
plt.show()

# Save the results to a CSV file
results_df.to_csv('sensitivity_analysis_results.csv', index=False)
print("Sensitivity analysis results saved to sensitivity_analysis_results.csv")
