import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor

# Konfigurasi Halaman
st.set_page_config(page_title="House Price Predictor DT", layout="wide")

# 1. Load Data
@st.cache_data
def get_data():
    df = pd.read_csv('house_prices.csv')
    # Pre-processing sederhana: ubah waterfront jadi angka
    if df['waterfront'].dtype == 'O':
        df['waterfront'] = df['waterfront'].map({'Y': 1, 'N': 0}).fillna(0)
    return df

df = get_data()

# 2. Training Model (Decision Tree)
# Kita pilih fitur yang paling relevan
features = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built', 'floors']
X = df[features]
y = df['price']

model = DecisionTreeRegressor(max_depth=12, random_state=42)
model.fit(X, y)

# 3. Sidebar untuk Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Prediksi Harga", "Analisis Model & Data"])

# --- HALAMAN 1: PREDIKSI ---
if page == "Prediksi Harga":
    st.title("🤖 Prediksi Harga Rumah (Decision Tree)")
    st.write("Sesuaikan parameter di bawah untuk melihat estimasi harga rumah.")

    col1, col2 = st.columns(2)
    
    with col1:
        sqft = st.slider("Luas Bangunan (sqft)", int(df.sqft_living.min()), int(df.sqft_living.max()), 2000)
        grade = st.slider("Grade (Kualitas)", 1, 13, 7)
        yr_built = st.number_input("Tahun Dibangun", 1900, 2025, 2000)

    with col2:
        beds = st.number_input("Jumlah Kamar Tidur", 1, 10, 3)
        baths = st.number_input("Jumlah Kamar Mandi", 0.5, 8.0, 2.5, step=0.25)
        floors = st.selectbox("Jumlah Lantai", sorted(df.floors.unique()), index=2)

    st.divider()
    
    # Tombol Prediksi
    if st.button("Hitung Estimasi Harga", use_container_width=True):
        input_data = np.array([[beds, baths, sqft, grade, yr_built, floors]])
        prediction = model.predict(input_data)[0]
        
        st.success(f"### Estimasi Harga Pasar: ${prediction:,.2f}")
        
        # Penjelasan singkat tentang Decision Tree
        st.info("💡 **Info Model:** Decision Tree membagi data ke dalam 'cabang' keputusan berdasarkan fitur di atas untuk menemukan kecocokan harga yang paling serupa.")

# --- HALAMAN 2: ANALISIS MODEL & DATA ---
else:
    st.title("📊 Analisis Model & Data")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Fitur Paling Berpengaruh")
        # Menghitung Feature Importance
        importance = pd.DataFrame({
            'Fitur': features,
            'Kepentingan': model.feature_importances_
        }).sort_values(by='Kepentingan', ascending=True)
        
        fig_imp = px.bar(importance, x='Kepentingan', y='Fitur', orientation='h', 
                         title="Faktor Penentu Harga (Decision Tree)")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.subheader("Distribusi Harga vs Grade")
        fig_scatter = px.scatter(df, x="grade", y="price", color="price",
                                 title="Semakin tinggi grade, harga cenderung eksponensial")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Sampel Dataset")
    st.write(df.head(10))