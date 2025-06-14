import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [2, 3, 4, 5, 6]
}
df = pd.DataFrame(data)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df)
df_MinMax = pd.DataFrame(data=scaled_features, columns=df.columns)

correlation_matrix = df_MinMax.corr()
plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.yticks(np.arange(len(correlation_matrix.index)), correlation_matrix.index)
plt.title('корреляционная матрица')
plt.tight_layout()
plt.show()
