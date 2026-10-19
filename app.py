import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Penghitung Omprengan", page_icon="🍛", layout="centered"
)

st.title("🍛 Penghitung Tumpukan Omprengan")
st.write(
    "Ambil foto tumpukan omprengan menggunakan kamera HP Anda, lalu sistem"
    " akan mendeteksi dan menghitung jumlahnya."
)

# Pilihan metode input kamera langsung dari HP
image_file = st.camera_input("Ambil Foto Tumpukan Omprengan")

if image_file is not None:
  # Konversi gambar dari Streamlit ke OpenCV format
  bytes_data = image_file.getvalue()
  np_arr = np.frombuffer(bytes_data, np.uint8)
  img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

  # Salin gambar untuk ditampilkan nanti
  output_img = img.copy()

  # 1. Pra-pemrosesan Gambar (Pre-processing)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Peningkatan kontras otomatis (Canny Edge)
  edged = cv2.Canny(blurred, 50, 150)

  # 2. Temukan Kontur / Garis Tepi
  contours, _ = cv2.findContours(
      edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  count = 0
  min_area = 500  # Sesuaikan ukuran minimal objek omprengan di foto
  max_area = 50000  # Sesuaikan ukuran maksimal objek

  for c in contours:
    area = cv2.contourArea(c)
    if min_area < area < max_area:
      perimeter = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

      if len(approx) >= 4:
        count += 1
        cv2.drawContours(output_img, [c], -1, (0, 255, 0), 3)

  # 3. Tampilkan Hasil
  st.success(f"### Perkiraan Total Omprengan: {count} buah")

  # Konversi warna kembali ke RGB untuk Streamlit
  output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
  st.image(
      output_img_rgb,
      caption="Hasil Deteksi (Garis Hijau)",
      use_container_width=True,
  )

  st.info(
      "Tips: Jika hasil kurang akurat, pastikan menggunakan alas background"
      " yang kontras (misal: hitam) dan pencahayaan yang cukup."
  )
