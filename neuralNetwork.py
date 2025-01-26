import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
import tensorflow

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential

df = pd.read_csv("summerOly_medal_counts.csv")
#print(df)

df2 = pd.read_csv("summerOly_hosts.csv")
#print(df2)

#list = ["Rank", "NOC", "Gold", "Silver", "Bronze", "Total", "Year"]
df_merged = df.merge(df2, on="Year", how='outer')
#print(df_merged)

encoder = OneHotEncoder(handle_unknown='ignore')
scaler = MinMaxScaler()
df_merged['Year'] = scaler.fit_transform(df_merged[['Year']])
categorical_features = encoder.fit_transform(df_merged[['Host', 'NOC']]).toarray()
X = np.hstack((df_merged[['Year']].values, categorical_features))
#X[:, 0] = scaler.fit_transform(X[:, 0].reshape(-1, 1)).flatten()
y = df_merged[['Gold', 'Silver', 'Bronze', 'Total']].values
#print(X.shape)
#print(y.shape)


model = Sequential()
# model.add(Dense(8, input_dim=3, activation='relu'))
# model.add(Dense(4))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))  # Increased hidden layer size for complex data
model.add(Dense(32, activation='relu'))
model.add(Dense(4))  # 4 outputs for regression
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# test_data = pd.get_dummies(pd.DataFrame([['2028', 'Los Angeles, United States', 'United States']],
#                                         columns=['Year', 'Host', 'NOC']))
# test_data['Year'] = scaler.transform(test_data[['Year']])
# test_data = test_data.reindex(columns=X.columns, fill_value=0)
# prediction = model.predict(test_data.values)
# print(f"Predicted medal counts: {prediction[0]}")

test_data = np.array([[2028]])
test_data = scaler.transform(test_data)


test_categorical = encoder.transform([['Los Angeles, United States', 'United States']]).toarray()
test_input = np.hstack((test_data, test_categorical))


prediction = model.predict(test_input)
print(f"Predicted medal counts: {prediction[0]}")

