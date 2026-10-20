import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Penghitung Omprengan", page_icon="🍱", layout="centered"
)

st.title("🍱 Penghitung Tumpukan Ompreng Stainless")
st.write(
    "Arahkan kamera HP ke tumpukan ompreng untuk menghitung jumlahnya secara"
    " otomatis."
)

# Ambil API Key dari Streamlit Secrets atau manual
try:
  API_KEY = st.secrets["MZ4OiGeGmiyc1nBZhfFv"]
except Exception:
  API_KEY = "rf_2eIgbWwEQBUjLLFvqecHQGqhS7Z2"


@st.cache_resource
def load_model():
  rf = Roboflow(api_key=API_KEY)
  # Project workspace ID & Model ID dari Roboflow kamu
  project = rf.workspace("pembagian-ompreng").project("penghitung-ompreng-1")
  model = project.version(1).model
  return model


# Ambil foto dari kamera HP
picture = st.camera_input("Ambil Foto Tumpukan Ompreng")

if picture is not None:
  # Simpan sementara gambar untuk dikirim ke Roboflow
  temp_img_path = "temp_ompreng.jpg"
  with open(temp_img_path, "wb") as f:
    f.write(picture.getvalue())

  with st.spinner("Sedang mendeteksi dan menghitung ompreng..."):
    try:
      model = load_model()
      # Lakukan prediksi
      prediction = model.predict(temp_img_path, confidence=40, overlap=30)

      # Simpan dan tampilkan gambar hasil prediksi
      prediction.save("prediction.jpg")
      result_json = prediction.json()

      predictions_list = result_json.get("predictions", [])
      jumlah_ompreng = len(predictions_list)

      st.success(
          f"🎉 Jumlah tumpukan ompreng terdeteksi: **{jumlah_ompreng} buah**"
      )

      # Tampilkan gambar bercak/box deteksi
      result_image = Image.open("prediction.jpg")
      st.image(
          result_image,
          caption="Hasil Deteksi Model AI",
          use_container_width=True,
      )

    except Exception as e:
      st.error(f"Gagal memproses gambar: {e}")
