# 🧪 Use Cases: Testing Retail Price Optimization App

## 📍 URL Aplikasi

**Live Demo**: https://retail-price-optimization-mugni.streamlit.app/

---

## 🎯 Use Case 1: Single Product Optimization

### Skenario

Seorang pricing analyst ingin mengoptimalkan harga untuk satu produk spesifik dalam kategori tertentu.

### Langkah-langkah

| Step | Action                                                     | Expected Result                                                |
| ---- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| 1    | Buka aplikasi Streamlit                                    | Dashboard tampil dengan header "AI Pricing Strategy Dashboard" |
| 2    | Di sidebar, pilih **Optimization Mode** = "Single Product" | Mode single product aktif                                      |
| 3    | Pilih kategori, misal: **"bed_bath_table"**                | Muncul info jumlah produk di kategori tersebut                 |
| 4    | Pilih produk dari dropdown (lihat harga dan score)         | Detail produk muncul di main area                              |
| 5    | Klik tombol **"🚀 Run AI Optimization"**                   | Loading spinner muncul, lalu hasil optimasi ditampilkan        |

### Expected Results

**Product Details Section:**

- ✅ Tampil harga saat ini (Base Price)
- ✅ Tampil freight cost
- ✅ Tampil product score
- ✅ Tampil harga 3 kompetitor
- ✅ Tampil market position (Above/Below average)

**Optimization Results Section:**

- ✅ **Current Price**: Harga saat ini (misal $50.00)
- ✅ **Optimal Price (AI)**: Rekomendasi harga baru dengan % change
- ✅ **Potential Revenue**: Estimasi revenue jika pakai harga optimal
- ✅ **Revenue Uplift**: Persentase kenaikan revenue

**Charts:**

- ✅ **Revenue Curve**: Grafik garis menunjukkan revenue di berbagai price point
- ✅ **Price Comparison Bar**: Perbandingan harga Anda vs kompetitor

**Strategic Insight:**

- ✅ Muncul rekomendasi: "INCREASE PRICE", "DECREASE PRICE", atau "MAINTAIN PRICE"
- ✅ Penjelasan mengapa rekomendasi tersebut diberikan

---

## 🎯 Use Case 2: Batch Optimization

### Skenario

Seorang category manager ingin mengoptimalkan harga semua produk dalam satu kategori sekaligus.

### Langkah-langkah

| Step | Action                                                         | Expected Result                        |
| ---- | -------------------------------------------------------------- | -------------------------------------- |
| 1    | Buka aplikasi Streamlit                                        | Dashboard tampil                       |
| 2    | Di sidebar, pilih **Optimization Mode** = "Batch Optimization" | Mode batch aktif                       |
| 3    | Pilih kategori, misal: **"computers_accessories"**             | Muncul info jumlah produk              |
| 4    | Klik tombol **"🚀 Run Batch Optimization"**                    | Progress bar muncul, optimasi berjalan |
| 5    | Tunggu hingga selesai                                          | Hasil batch optimization ditampilkan   |

### Expected Results

**Summary Metrics:**

- ✅ **Total Products**: Jumlah produk yang dioptimasi
- ✅ **Recommend Increase**: Berapa produk yang harus naikkan harga
- ✅ **Recommend Decrease**: Berapa produk yang harus turunkan harga
- ✅ **Total Revenue Uplift**: Total potensi kenaikan revenue dalam $

**Results Table:**

- ✅ Tabel dengan kolom:
  - Product ID
  - Current Price
  - Optimal Price
  - Price Change %
  - Current Revenue
  - Optimal Revenue
  - Revenue Uplift %
  - Action (Increase/Decrease/Maintain)

**Charts:**

- ✅ **Histogram**: Distribusi rekomendasi perubahan harga
- ✅ **Pie Chart**: Proporsi action (Increase vs Decrease vs Maintain)

**Export:**

- ✅ Tombol **"📄 Download CSV"** - download hasil dalam format CSV
- ✅ Tombol **"📊 Download Excel"** - download hasil dalam format XLSX

---

## 🎯 Use Case 3: Advanced Settings

### Skenario

User ingin menyesuaikan range harga yang diuji.

### Langkah-langkah

| Step | Action                                             | Expected Result                                                 |
| ---- | -------------------------------------------------- | --------------------------------------------------------------- |
| 1    | Di sidebar, expand **"⚙️ Advanced Settings"**      | Panel settings terbuka                                          |
| 2    | Geser slider **"Price Range (+/-)"** ke 0.30 (30%) | Nilai slider berubah menjadi 30%                                |
| 3    | Jalankan optimasi                                  | Sistem akan test harga dari -30% sampai +30% dari harga current |

### Expected Results

- ✅ Revenue curve lebih lebar (lebih banyak price points diuji)
- ✅ Optimal price mungkin berbeda karena range lebih besar

---

## 🎯 Use Case 4: Error Handling

### Skenario

Testing bahwa aplikasi menangani error dengan baik.

### Test Cases

| Scenario              | Expected Behavior                                               |
| --------------------- | --------------------------------------------------------------- |
| Kategori tanpa produk | Warning message "No products found in this category"            |
| Model gagal predict   | Error message ditampilkan dengan detail                         |
| Data kosong           | "Critical Error: Could not load data or model" dan app berhenti |

---

## ✅ Test Checklist

### Single Product Optimization

- [ ] Bisa pilih kategori dari dropdown
- [ ] Bisa pilih produk spesifik
- [ ] Detail produk tampil dengan benar
- [ ] Harga kompetitor tampil
- [ ] Tombol "Run AI Optimization" berfungsi
- [ ] Revenue curve chart muncul
- [ ] Price comparison chart muncul
- [ ] Rekomendasi strategic insight muncul
- [ ] Nilai optimal price masuk akal

### Batch Optimization

- [ ] Progress bar muncul saat optimasi berjalan
- [ ] Semua produk dalam kategori dioptimasi
- [ ] Summary metrics tampil dengan benar
- [ ] Results table tampil dengan data lengkap
- [ ] Distribution histogram muncul
- [ ] Pie chart action muncul
- [ ] Download CSV berfungsi
- [ ] Download Excel berfungsi
- [ ] File yang didownload berisi data yang benar

### UI/UX

- [ ] Sidebar navigation berfungsi
- [ ] Dark mode styling konsisten
- [ ] Charts responsive (bisa di-hover, zoom)
- [ ] Loading states informative
- [ ] No layout breaking pada berbagai screen size

---

## 📝 Sample Test Data

Untuk testing manual, gunakan data berikut:

**Kategori dengan banyak produk:**

- `bed_bath_table`
- `computers_accessories`
- `health_beauty`

**Kategori dengan sedikit produk:**

- `fashion_sport`
- `audio`

---

## 🔄 Regression Test

Setelah setiap update, pastikan:

1. Single product optimization masih berfungsi
2. Batch optimization masih berfungsi
3. Export CSV/Excel masih berfungsi
4. Charts masih render dengan benar
5. Tidak ada error di console/logs
