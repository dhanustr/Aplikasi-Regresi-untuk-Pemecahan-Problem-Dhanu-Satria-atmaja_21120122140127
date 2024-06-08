import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data dari file CSV
urdata = r"C:\Users\Public\python\Student_Performance.csv"
data = pd.read_csv(urdata)

# Menampilkan beberapa data untuk memastikan pembacaan berhasil
print(data.head())

# Mengambil kolom yang diperlukan
data = data [['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]
X = data[['Sample Question Papers Practiced']].values.reshape (-1, 1)
y = data['Performance Index'].values

linear_model = LinearRegression() 
linear_model.fit(X, y) 
y_pred_linear = linear_model.predict(X) 
plt.scatter(X, y, color='blue', label='Data Points') 
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('Sample Question Papers Practiced') 
plt.ylabel('Performance Index')
plt.title('Linear Regression')

plt.legend() 
plt.show() 
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear)) 
print(f'RMS Error for Linear Model: {rms_linear}')