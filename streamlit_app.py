import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# === SETTING HALAMAN ===
st.set_page_config(page_title="Final Project Data Science", layout="wide")
st.title("🔍 Klasifikasi Tipe Karyawan Berdasarkan Karakter Kerja")

# === STYLING ===
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .stButton > button {
        background-color: #0e6dfd;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# === LOAD MODEL ===
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

label_map = {
    0: 'Karyawan Stabil',
    1: 'Karyawan Ambisius',
    2: 'Karyawan Kolaboratif',
    3: 'Karyawan Analitis'
}

features = [
    'WorkLifeBalance', 'OverTime', 'YearsWithCurrManager',
    'TrainingTimesLastYear', 'JobInvolvement', 'DistanceFromHome',
    'TotalWorkingYears', 'RelationshipSatisfaction', 'EnvironmentSatisfaction'
]

# === CREATE TEMPLATE CSV ===
def create_template():
    template_data = {
        'WorkLifeBalance': [3, 4, 2, 3],
        'OverTime': ['No', 'Yes', 'No', 'Yes'],
        'YearsWithCurrManager': [2, 1, 3, 4],
        'TrainingTimesLastYear': [2, 3, 1, 4],
        'JobInvolvement': [3, 4, 2, 3],
        'DistanceFromHome': [10, 5, 15, 8],
        'TotalWorkingYears': [5, 3, 7, 10],
        'RelationshipSatisfaction': [4, 3, 2, 4],
        'EnvironmentSatisfaction': [3, 4, 3, 2]
    }
    return pd.DataFrame(template_data)

# === PREDIKSI INDIVIDU ===
st.subheader("🧍 Prediksi Individu")

with st.form("form_input"):
    col1, col2 = st.columns(2)
    with col1:
        work_life = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])
        overtime = st.selectbox("Lembur?", ["Yes", "No"])
        years_with_mgr = st.slider("Tahun dengan Manajer", 0, 40, 5)
        training = st.slider("Pelatihan Tahun Ini", 0, 10, 3)
        job_involve = st.selectbox("Keterlibatan Kerja (1-4)", [1, 2, 3, 4])
    with col2:
        distance = st.slider("Jarak dari Rumah (km)", 1, 50, 10)
        total_years = st.slider("Total Tahun Bekerja", 0, 40, 7)
        rel_sat = st.selectbox("Kepuasan Relasi (1-4)", [1, 2, 3, 4])
        env_sat = st.selectbox("Kepuasan Lingkungan (1-4)", [1, 2, 3, 4])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    overtime_val = 1 if overtime.lower() == "yes" else 0
    input_data = np.array([[work_life, overtime_val, years_with_mgr, training,
                            job_involve, distance, total_years, rel_sat, env_sat]])
    scaled_input = scaler.transform(input_data)
    pred_cluster = kmeans.predict(scaled_input)[0]
    archetype = label_map[pred_cluster]

    st.success(f"Hasil klasifikasi: **{archetype}**")

# === UPLOAD FILE CSV ===
st.subheader("📁 Prediksi Batch dari File CSV")

# Add template download button
template_df = create_template()
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Dataset CSV",
    data=csv_template,
    file_name='employee_template.csv',
    mime='text/csv',
    help="Download template file untuk input data karyawan"
)

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file:
    try:
        df_batch = pd.read_csv(uploaded_file)

        missing_cols = [col for col in features if col not in df_batch.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            st.stop()

        df_batch['OverTime'] = df_batch['OverTime'].str.lower().map({'yes': 1, 'no': 0})
        X_scaled = scaler.transform(df_batch[features])
        cluster_labels = kmeans.predict(X_scaled)
        archetypes = [label_map[c] for c in cluster_labels]
        df_batch['Archetype'] = archetypes

        st.success("✅ Batch berhasil diklasifikasikan")
        st.dataframe(df_batch)

        # === VISUALISASI PCA ===
        st.subheader("📊 Visualisasi PCA 2D")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_batch['PCA1'] = X_pca[:, 0]
        df_batch['PCA2'] = X_pca[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_batch, x='PCA1', y='PCA2', hue='Archetype', palette='Set2', s=60, edgecolor='black')
        plt.title("Visualisasi Tipe Karyawan (PCA)")
        plt.xlabel("PCA Komponen 1")
        plt.ylabel("PCA Komponen 2")
        st.pyplot(fig)

        # === PIE CHART DISTRIBUSI ===
        st.subheader("🧩 Distribusi Tipe Karyawan")
        dist = df_batch['Archetype'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(dist, labels=dist.index, autopct='%1.1f%%', startangle=140)
        st.pyplot(fig2)

        # === FILTER DAN DOWNLOAD ===
        st.subheader("🔎 Filter dan Unduh CSV")
        selected_type = st.selectbox("Pilih Tipe", df_batch['Archetype'].unique())
        filtered_df = df_batch[df_batch['Archetype'] == selected_type]
        st.dataframe(filtered_df)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Unduh CSV Hasil Filter",
            data=csv,
            file_name=f'hasil_{selected_type}.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Gagal memproses file: {e}")