import numpy as np
import pandas as pd
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
print(df_merged)

X = df_merged[['Year', 'Host', 'NOC']].values
y = df_merged[['Gold', 'Silver', 'Bronze', 'Total']].values

model = Sequential()
model.add(Dense(8, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=1, verbose=1)

test_data = np.array([['2028', 'Los Angeles, United States', 'United States']])

prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
print(f"Predicted label: {predicted_label[0][0]}")

