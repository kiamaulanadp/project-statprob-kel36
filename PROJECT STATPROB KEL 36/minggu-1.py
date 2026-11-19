
# ============================================================
# PROJECT STATISTIKA DAN PROBABILITAS
# KELOMPOK 36 - TEMA D: OPERASIONAL DAN PENJUALAN UMKM KOPI
# MINGGU 1: EKSPLORASI DAN PEMBERSIHAN DATA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. PERSIAPAN FOLDER
# ============================================================

FILE = "dataset/Kelompok_36_Tema_D_UMKM_Kopi.csv"

FOLDER_OUTPUT = "output"
FOLDER_GRAFIK = "output/grafik"

os.makedirs(FOLDER_OUTPUT, exist_ok=True)
os.makedirs(FOLDER_GRAFIK, exist_ok=True)

print("=" * 65)
print("PROJECT STATISTIKA DAN PROBABILITAS")
print("KELOMPOK 36 - TEMA D: UMKM KOPI")
print("ANALISIS MINGGU 1")
print("=" * 65)


# ============================================================
# 2. MEMBACA DATASET CSV
# ============================================================

df = pd.read_csv(FILE)

print("\n[1] INFORMASI DATASET")

print("Jumlah observasi :", len(df))
print("Jumlah variabel  :", len(df.columns))

print("\nNama variabel:")
print(df.columns.tolist())

print("\nLima data pertama:")
print(df.head())


# Menentukan variabel numerik
numeric_cols = [
    "Jumlah_Pengunjung",
    "Suhu_Rata2",
    "Biaya_Operasional_Ribu",
    "Omzet_Harian_Ribu"
]

# Memastikan variabel numerik memiliki tipe data yang benar
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 3. IDENTIFIKASI SKALA PENGUKURAN
# ============================================================

print("\n[2] IDENTIFIKASI SKALA PENGUKURAN")

skala = pd.DataFrame([
    ["ID_Shift", "Identitas observasi", "Nominal"],
    ["Shift_Operasional", "Kategori waktu operasional", "Nominal"],
    ["Kondisi_Cuaca", "Kategori kondisi cuaca", "Nominal"],
    ["Promo_Aktif", "Status promosi", "Nominal"],
    ["Metode_Bayar_Dominan", "Kategori pembayaran", "Nominal"],
    ["Jumlah_Pengunjung", "Jumlah pengunjung", "Rasio"],
    ["Suhu_Rata2", "Suhu rata-rata dalam Celsius", "Interval"],
    ["Biaya_Operasional_Ribu", "Biaya operasional", "Rasio"],
    ["Omzet_Harian_Ribu", "Omzet harian", "Rasio"]
], columns=[
    "Variabel",
    "Keterangan",
    "Skala_Pengukuran"
])

print(skala.to_string(index=False))


# ============================================================
# 4. IDENTIFIKASI MISSING VALUES
# ============================================================

print("\n[3] IDENTIFIKASI MISSING VALUES")

missing = pd.DataFrame({
    "Variabel": df.columns,
    "Jumlah_Missing": [
        df[col].isna().sum()
        for col in df.columns
    ],
    "Persentase_Missing": [
        df[col].isna().mean() * 100
        for col in df.columns
    ]
})

print(missing.to_string(index=False))

total_missing = df.isna().sum().sum()

print("\nTotal missing values:", total_missing)


# ============================================================
# 5. PENANGANAN MISSING VALUES
# ============================================================

print("\n[4] PENANGANAN MISSING VALUES")

# Menyimpan salinan dataset asli
data_asli = df.copy()

# Membuat salinan untuk proses pembersihan
data_bersih = df.copy()

hasil_imputasi = []

for col in numeric_cols:

    jumlah_kosong = data_bersih[col].isna().sum()

    if jumlah_kosong > 0:

        nilai_median = data_bersih[col].median()

        data_bersih[col] = data_bersih[col].fillna(
            nilai_median
        )

        hasil_imputasi.append({
            "Variabel": col,
            "Jumlah_Data_Diisi": jumlah_kosong,
            "Metode": "Median",
            "Nilai_Imputasi": nilai_median
        })

imputasi = pd.DataFrame(hasil_imputasi)

print(imputasi.to_string(index=False))

print("\nTotal missing setelah imputasi:",
      data_bersih.isna().sum().sum())


# ============================================================
# 6. DETEKSI OUTLIER MENGGUNAKAN METODE IQR
# ============================================================

print("\n[5] DETEKSI OUTLIER MENGGUNAKAN IQR")

data_analisis = data_bersih.copy()

hasil_outlier = []

batas_iqr = {}

for col in numeric_cols:

    Q1 = data_bersih[col].quantile(0.25)
    Q3 = data_bersih[col].quantile(0.75)

    IQR = Q3 - Q1

    batas_bawah = Q1 - (1.5 * IQR)
    batas_atas = Q3 + (1.5 * IQR)

    batas_iqr[col] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Batas_Bawah": batas_bawah,
        "Batas_Atas": batas_atas
    }

    # Mengidentifikasi data yang berada di luar pagar IQR
    kondisi_outlier = (
        (data_bersih[col] < batas_bawah) |
        (data_bersih[col] > batas_atas)
    )

    jumlah_outlier = kondisi_outlier.sum()

    print(f"\nVariabel: {col}")
    print(f"Q1            : {Q1:.2f}")
    print(f"Q3            : {Q3:.2f}")
    print(f"IQR           : {IQR:.2f}")
    print(f"Batas bawah   : {batas_bawah:.2f}")
    print(f"Batas atas    : {batas_atas:.2f}")
    print(f"Jumlah outlier: {jumlah_outlier}")

    for index in data_bersih.index[kondisi_outlier]:

        hasil_outlier.append({
            "ID_Shift": data_bersih.loc[index, "ID_Shift"],
            "Variabel": col,
            "Nilai_Asli": data_bersih.loc[index, col],
            "Batas_Bawah": batas_bawah,
            "Batas_Atas": batas_atas
        })

    # Penanganan outlier menggunakan winsorisasi
    # Nilai ekstrem dibatasi pada pagar IQR
    data_analisis[col] = data_bersih[col].clip(
        lower=batas_bawah,
        upper=batas_atas
    )

outlier = pd.DataFrame(hasil_outlier)

print("\nTABEL DATA OUTLIER")

if len(outlier) > 0:
    print(outlier.to_string(index=False))
else:
    print("Tidak ditemukan outlier.")


# ============================================================
# 7. STATISTIKA DESKRIPTIF
# ============================================================

print("\n[6] STATISTIKA DESKRIPTIF")

hasil_statistik = []

for col in numeric_cols:

    nilai = data_analisis[col]

    Q1 = nilai.quantile(0.25)
    Q3 = nilai.quantile(0.75)

    hasil_statistik.append({

        "Variabel": col,

        "Jumlah_Data": nilai.count(),

        "Mean": nilai.mean(),

        "Median": nilai.median(),

        "Modus": ", ".join(
            map(str, nilai.mode().tolist())
        ),

        "Standar_Deviasi": nilai.std(ddof=1),

        "Varians": nilai.var(ddof=1),

        "IQR": Q3 - Q1,

        "Skewness": nilai.skew()

    })

statistik = pd.DataFrame(hasil_statistik)

print(
    statistik.round(3).to_string(index=False)
)


# ============================================================
# 8. GRAFIK 1 - HISTOGRAM
# ============================================================

print("\n[7] MEMBUAT HISTOGRAM")

plt.figure(figsize=(9, 5))

plt.hist(
    data_analisis["Omzet_Harian_Ribu"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribusi Omzet Harian UMKM Kopi")

plt.xlabel("Omzet Harian (Ribu Rupiah)")

plt.ylabel("Frekuensi")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FOLDER_GRAFIK + "/01_histogram_omzet.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. GRAFIK 2 - BOXPLOT SEBELUM DAN SESUDAH
# ============================================================

print("[8] MEMBUAT BOXPLOT")

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

# Boxplot sebelum penanganan outlier
axes[0].boxplot(
    data_bersih["Omzet_Harian_Ribu"],
    showmeans=True
)

axes[0].set_title(
    "Sebelum Penanganan Outlier"
)

axes[0].set_ylabel(
    "Omzet (Ribu Rupiah)"
)

# Boxplot setelah penanganan outlier
axes[1].boxplot(
    data_analisis["Omzet_Harian_Ribu"],
    showmeans=True
)

axes[1].set_title(
    "Sesudah Penanganan Outlier"
)

axes[1].set_ylabel(
    "Omzet (Ribu Rupiah)"
)

plt.suptitle(
    "Perbandingan Boxplot Omzet Harian"
)

plt.tight_layout()

plt.savefig(
    FOLDER_GRAFIK + "/02_boxplot_omzet.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. GRAFIK 3 - GROUPED BAR CHART
# ============================================================

print("[9] MEMBUAT GROUPED BAR CHART")

grouped = data_bersih.groupby(
    ["Kondisi_Cuaca", "Promo_Aktif"]
)["Jumlah_Pengunjung"].mean().unstack()

# Menampilkan kategori cuaca sesuai urutan
urutan_cuaca = [
    "Cerah",
    "Mendung",
    "Hujan"
]

grouped = grouped.reindex(
    [x for x in urutan_cuaca if x in grouped.index]
)

ax = grouped.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title(
    "Rata-rata Pengunjung Berdasarkan Cuaca dan Promo"
)

ax.set_xlabel(
    "Kondisi Cuaca"
)

ax.set_ylabel(
    "Rata-rata Jumlah Pengunjung"
)

ax.legend(
    title="Promo Aktif"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FOLDER_GRAFIK + "/03_grouped_bar.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. MENYIMPAN HASIL ANALISIS KE EXCEL
# ============================================================

print("\n[10] MENYIMPAN HASIL ANALISIS")

FILE_OUTPUT = "output/hasil_analisis_minggu_1.xlsx"

with pd.ExcelWriter(FILE_OUTPUT) as writer:

    data_asli.to_excel(
        writer,
        sheet_name="Data_Asli",
        index=False
    )

    data_bersih.to_excel(
        writer,
        sheet_name="Data_Imputasi",
        index=False
    )

    data_analisis.to_excel(
        writer,
        sheet_name="Data_Analisis",
        index=False
    )

    skala.to_excel(
        writer,
        sheet_name="Skala_Pengukuran",
        index=False
    )

    missing.to_excel(
        writer,
        sheet_name="Missing_Values",
        index=False
    )

    imputasi.to_excel(
        writer,
        sheet_name="Imputasi_Median",
        index=False
    )

    outlier.to_excel(
        writer,
        sheet_name="Deteksi_Outlier",
        index=False
    )

    statistik.to_excel(
        writer,
        sheet_name="Statistik_Deskriptif",
        index=False
    )

    grouped.reset_index().to_excel(
        writer,
        sheet_name="Grouped_Bar_Data",
        index=False
    )

# ============================================================
# 12. RINGKASAN AKHIR
# ============================================================

print("\n" + "=" * 65)

print("ANALISIS MINGGU 1 BERHASIL DIJALANKAN!")

print("=" * 65)

print("Jumlah observasi       :", len(data_asli))
print("Jumlah variabel        :", len(data_asli.columns))
print("Total missing awal     :", total_missing)
print("Total missing akhir    :", data_bersih.isna().sum().sum())
print("Jumlah outlier terdeteksi:", len(outlier))

print("\nFile hasil Excel:")
print(FILE_OUTPUT)

print("\nFolder grafik:")
print(FOLDER_GRAFIK)

print("\nSELURUH PROSES SELESAI!")