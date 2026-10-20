import cv2
import numpy as np
import streamlit as st
from inference import get_model
from PIL import Image

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Penghitung Omprengan Real-Time", page_icon="🍱", layout="centered"
)

st.title("🍱 Penghitung Tumpukan Ompreng Stainless")
st.write(
    "Arahkan kamera HP ke tumpukan ompreng untuk menghitung jumlahnya secara"
    " otomatis."
)

# Konfigurasi Model ID dan API Key
MODEL_ID = "penghitung-ompreng-1-rfdetr-small-t1"

# Mengambil API Key secara aman dari st.secrets Streamlit
try:
  API_KEY = st.secrets["ROBOFLOW_API_KEY"]
except Exception:
  # Fallback jika dijalankan lokal (ubah string di bawah dengan API Key kamu jika perlu)
  API_KEY = "MASUKKAN_API_KEY_KAMU_DISINI"


@st.cache_resource
def load_roboflow_model():
  return get_model(model_id=MODEL_ID, api_key=API_KEY)


# Memuat model AI
try:
  model = load_roboflow_model()
except Exception as e:
  st.error(f"Gagal memuat model Roboflow: {e}")

# Ambil gambar menggunakan kamera HP
picture = st.camera_input("Ambil Foto Tumpukan Ompreng")

if picture is not None:
  # Konversi gambar dari Streamlit ke format OpenCV
  bytes_data = picture.getvalue()
  cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

  with st.spinner("Sedang mendeteksi dan menghitung ompreng..."):
    # Jalankan inferensi ke model Roboflow
    results = model.infer(cv_img)[0]
    predictions = results.predictions

    # Hitung total objek ompreng yang terdeteksi
    jumlah_ompreng = len(predictions)

    # Tampilkan hasil perhitungan
    st.success(
        f"🎉 Jumlah tumpukan ompreng terdeteksi: **{jumlah_ompreng} buah**"
    )

    # Visualisasikan kotak hasil deteksi pada gambar
    annotated_image = results.plot()
    st.image(
        annotated_image[:, :, ::-1],
        caption="Hasil Deteksi Model AI",
        use_container_width=True,
    )
