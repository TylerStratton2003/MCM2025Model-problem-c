import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
import tensorflow
import csv

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential

df = pd.read_csv("summerOly_medal_counts.csv")

df2 = pd.read_csv("summerOly_hosts.csv")

df_merged = df.merge(df2, on="Year", how='outer')
df_merged = df_merged.dropna()
print(df_merged)

encoder = OneHotEncoder(handle_unknown='ignore')
scaler = MinMaxScaler()
df_merged['Year'] = scaler.fit_transform(df_merged[['Year']])
categorical_features = encoder.fit_transform(df_merged[['Host', 'NOC']]).toarray()
X = np.hstack((df_merged[['Year']].values, categorical_features))
y = df_merged[['Gold', 'Silver', 'Bronze', 'Total']].values
print(np.isnan(X).any())
print(np.isnan(y).any())


model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4))  # 4 outputs for regression
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X, y, epochs=50, batch_size=16, verbose=1)
test_data = np.array([[2028]])
test_data = scaler.transform(test_data)

countries_2024 = [
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

results_file = open('results_file.csv', 'w')
header = ['NOC', 'Gold', 'Silver', 'Bronze', 'Total']
writer = csv.DictWriter(results_file, fieldnames = header)
writer.writeheader()
for i in countries_2024:
    test_categorical = encoder.transform([['Los Angeles, United States', i]]).toarray()
    test_input = np.hstack((test_data, test_categorical))
    prediction = model.predict(test_input)
    print(f"Predicted medal counts for {i}: {prediction[0]}")
    writer.writerow({'NOC' : i, 'Gold' : prediction[0][0], 'Silver' : prediction[0][1], 'Bronze' : prediction[0][2], 'Total' : prediction[0][3]})


