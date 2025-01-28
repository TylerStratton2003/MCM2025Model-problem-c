import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tf_keras.models import Sequential
from tf_keras.layers import Dense
import csv


def load_and_merge_data(medal_counts_path, hosts_path):
    """Load and merge the medal counts and hosts data."""
    df = pd.read_csv(medal_counts_path)
    df2 = pd.read_csv(hosts_path)
    df_merged = df.merge(df2, on="Year", how="outer").dropna()
    return df_merged


def preprocess_data(df):
    """Preprocess data: scale numerical features and encode categorical ones."""
    encoder = OneHotEncoder(handle_unknown='ignore')
    scaler = MinMaxScaler()

    # Scale the 'Year' column
    df['Year'] = scaler.fit_transform(df[['Year']])

    # One-hot encode categorical columns
    categorical_features = encoder.fit_transform(df[['Host', 'NOC']]).toarray()

    # Combine scaled numerical and encoded categorical features
    X = np.hstack((df[['Year']].values, categorical_features))
    y = df[['Gold', 'Silver', 'Bronze', 'Total']].values

    return X, y, encoder, scaler


def build_model(input_dim):
    """Build the neural network model."""
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4))  # Output layer for Gold, Silver, Bronze, Total
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


def train_model(model, X, y, epochs=50, batch_size=16):
    """Train the model."""
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def predict_medals(model, encoder, scaler, year, host, countries):
    """Predict medal counts for a given year, host, and countries."""
    results = []
    test_data = np.array([[year]])
    test_data = scaler.transform(test_data)

    for country in countries:
        test_categorical = encoder.transform([[host, country]]).toarray()
        test_input = np.hstack((test_data, test_categorical))
        prediction = model.predict(test_input)
        results.append({
            'NOC': country,
            'Gold': prediction[0][0],
            'Silver': prediction[0][1],
            'Bronze': prediction[0][2],
            'Total': prediction[0][3]
        })

    return results


def save_predictions_to_csv(results, output_file):
    """Save predictions to a CSV file."""
    with open(output_file, 'w', newline='') as results_file:
        header = ['NOC', 'Gold', 'Silver', 'Bronze', 'Total']
        writer = csv.DictWriter(results_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    # File paths
    medal_counts_path = "summerOly_medal_counts.csv"
    hosts_path = "summerOly_hosts.csv"
    output_file = "results_file.csv"

    # Load and preprocess the data
    df = load_and_merge_data(medal_counts_path, hosts_path)
    X, y, encoder, scaler = preprocess_data(df)

    # Build and train the model
    model = build_model(input_dim=X.shape[1])
    model = train_model(model, X, y, epochs=50, batch_size=16)

    # Predict medals for the 2028 Olympics
    test_year = 2028
    host_city = "Los Angeles, United States"
    countries = [
        "United States", "China", "Japan", "Germany", "Australia"  # Add more countries as needed
    ]

    predictions = predict_medals(model, encoder, scaler, test_year, host_city, countries)

    # Save predictions to CSV
    save_predictions_to_csv(predictions, output_file)
    print(f"Predictions saved to {output_file}")
