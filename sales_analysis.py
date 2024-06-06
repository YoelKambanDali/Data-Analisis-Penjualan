# Import library yang diperlukan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Langkah 1: Pengumpulan Data
data = pd.read_csv('sales_data.csv')

# Langkah 2: Pembersihan Data
# Tidak diperlukan dalam contoh ini karena data sudah bersih

# Langkah 3: Transformasi Data
# Tidak diperlukan dalam contoh ini karena data sudah dalam format yang sesuai

# Langkah 4: Exploratory Data Analysis (EDA)

# Diagram 1: Bar Plot Pendapatan Tahun Ini di Tiap Cabang (warna: cyan)
plt.figure(figsize=(10, 6))
sns.barplot(x='Cabang', y='Pendapatan Tahun Ini (USD)', data=data, palette='viridis')
plt.title('Distribusi Pendapatan Tahun Ini di Tiap Cabang')
plt.xlabel('Cabang')
plt.ylabel('Pendapatan Tahun Ini (USD)')
plt.xticks(rotation=45)
plt.show()

# Diagram 2: Scatter Plot Jumlah Produk Terjual vs. Pendapatan Rata-rata per Produk (warna: orange)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Jumlah Produk Terjual', y='Pendapatan Rata-rata per Produk (USD)', data=data, color='orange')
plt.title('Hubungan Jumlah Produk Terjual dengan Pendapatan Rata-rata per Produk')
plt.xlabel('Jumlah Produk Terjual')
plt.ylabel('Pendapatan Rata-rata per Produk (USD)')
plt.show()

# Langkah 5: Pemodelan Data
# Memisahkan fitur (X) dan target (y)
X = data[['Pendapatan Tahun Lalu (USD)', 'Jumlah Karyawan', 'Total Pelanggan', 'Rating Kepuasan Pelanggan (Skala 1-10)', 'Jumlah Produk Terjual']]
y = data['Pendapatan Tahun Ini (USD)']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Langkah 6: Validasi dan Penyetelan Model
# Menggunakan mean squared error untuk mengukur performa model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Langkah 7: Interpretasi dan Penyajian Hasil
# Tidak diperlukan dalam contoh ini karena hasil sudah disajikan melalui MSE

# Diagram 3: Histogram Rating Kepuasan Pelanggan (warna: purple)
plt.figure(figsize=(8, 6))
sns.histplot(data['Rating Kepuasan Pelanggan (Skala 1-10)'], color='purple', bins=10)
plt.title('Distribusi Rating Kepuasan Pelanggan')
plt.xlabel('Rating Kepuasan Pelanggan (Skala 1-10)')
plt.ylabel('Frekuensi')
plt.show()

# Diagram 4: Box Plot Pendapatan Tahun Ini di Tiap Cabang (warna: green)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cabang', y='Pendapatan Tahun Ini (USD)', data=data, palette='Set3')
plt.title('Distribusi Pendapatan Tahun Ini di Tiap Cabang')
plt.xlabel('Cabang')
plt.ylabel('Pendapatan Tahun Ini (USD)')
plt.xticks(rotation=45)
plt.show()

# Diagram 5: Pie Chart Persentase Jumlah Karyawan di Setiap Cabang (warna: pink)
plt.figure(figsize=(8, 6))
colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightgrey']
explode = (0.1, 0, 0, 0, 0)  # Pemisahan slice pertama
data['Jumlah Karyawan'].groupby(data['Cabang']).sum().plot(kind='pie', autopct='%1.1f%%', colors=colors, explode=explode)
plt.title('Persentase Jumlah Karyawan di Setiap Cabang')
plt.ylabel('')
plt.show()
