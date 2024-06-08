import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Membaca data dari file CSV
urdata = r"C:\Users\Public\python\Student_Performance.csv"
data = pd.read_csv(urdata)

# Menampilkan beberapa data untuk memastikan pembacaan berhasil
print(data.head())

# Mengambil kolom yang diperlukan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]
X = data[['Sample Question Papers Practiced']].values
y = data['Performance Index'].values

# Regresi berbasis pangkat sederhana
X_pangkat = np.power(X, 2)  # Mengubah fitur masukan menjadi pangkat dua

# Memperkirakan parameter regresi secara langsung
A = np.column_stack([X_pangkat, np.ones_like(X_pangkat)])
params, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

# Membuat prediksi berdasarkan parameter yang ditemukan
y_pred_pangkat = np.dot(A, params)

# Plot hasil regresi
plt.scatter(X, y, color='blue', label='Data Points')  # Plot data
plt.plot(X, y_pred_pangkat, color='red', label='Power Regression')  # Plot hasil regresi pangkat sederhana
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.title('Power Regression')
plt.legend()
plt.show()

# Menghitung galat RMS
rms_pangkat_sederhana = np.sqrt(mean_squared_error(y, y_pred_pangkat))
print(f'RMS Error for Power Regression: {rms_pangkat_sederhana}')
