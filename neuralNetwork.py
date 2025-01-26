import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow

from tensorflow.python.keras.layers import Input, Dense

df = pd.read_csv("summerOly_medal_counts.csv")
#print(df)

df2 = pd.read_csv("summerOly_hosts.csv")
#print(df2)

#list = ["Rank", "NOC", "Gold", "Silver", "Bronze", "Total", "Year"]
df_merged = df.merge(df2, on="Year", how='outer')
print(df_merged)


