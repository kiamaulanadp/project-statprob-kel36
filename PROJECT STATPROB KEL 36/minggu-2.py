
# ============================================================
# PROJECT STATISTIKA DAN PROBABILITAS
# KELOMPOK 36 - TEMA D: UMKM KOPI
# MINGGU 2: PEMODELAN PROBABILITAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, binom
import os

# ============================================================
# 1. PERSIAPAN
# ============================================================

FILE = "dataset/Kelompok_36_Tema_D_UMKM_Kopi.csv"

os.makedirs("output", exist_ok=True)
os.makedirs("output/grafik", exist_ok=True)

print("=" * 65)
print("PROJECT STATISTIKA DAN PROBABILITAS")
print("KELOMPOK 36 - TEMA D: UMKM KOPI")
print("ANALISIS MINGGU 2: PEMODELAN PROBABILITAS")
print("=" * 65)


# ============================================================
# 2. MEMBACA DATASET
# ============================================================

df = pd.read_csv(FILE)

df["Biaya_Operasional_Ribu"] = pd.to_numeric(
    df["Biaya_Operasional_Ribu"],
    errors="coerce"
)

print("\nINFORMASI DATASET")
print("Jumlah observasi:", len(df))
print("Jumlah variabel:", len(df.columns))


# ============================================================
# 3. PELUANG EMPIRIS
# ============================================================

print("\n" + "=" * 65)
print("BAGIAN 1: PELUANG EMPIRIS")
print("=" * 65)

N = len(df)

# Kejadian A: Promo aktif
A = (df["Promo_Aktif"] == "Ya").sum()

# Kejadian B: Pembayaran dominan QRIS
B = (df["Metode_Bayar_Dominan"] == "QRIS").sum()

# Kejadian C: Kondisi cuaca hujan
C = (df["Kondisi_Cuaca"] == "Hujan").sum()

# Menghitung peluang empiris
P_A = A / N
P_B = B / N
P_C = C / N

print("\n1. Peluang Promo Aktif")
print(f"P(A) = {A}/{N} = {P_A:.4f}")
print(f"Persentase = {P_A*100:.2f}%")

print("\n2. Peluang Pembayaran QRIS")
print(f"P(B) = {B}/{N} = {P_B:.4f}")
print(f"Persentase = {P_B*100:.2f}%")

print("\n3. Peluang Cuaca Hujan")
print(f"P(C) = {C}/{N} = {P_C:.4f}")
print(f"Persentase = {P_C*100:.2f}%")


# ============================================================
# 4. PELUANG BERSYARAT
# ============================================================

print("\n" + "=" * 65)
print("BAGIAN 2: PELUANG BERSYARAT")
print("=" * 65)

# P(Promo Aktif | Pembayaran QRIS)

jumlah_promo_qris = (
    (df["Promo_Aktif"] == "Ya") &
    (df["Metode_Bayar_Dominan"] == "QRIS")
).sum()

P_A_given_B = jumlah_promo_qris / B

print("\n1. Peluang Promo Aktif jika Pembayaran QRIS")
print(f"P(A|B) = {jumlah_promo_qris}/{B}")
print(f"P(A|B) = {P_A_given_B:.4f}")
print(f"Persentase = {P_A_given_B*100:.2f}%")


# P(Promo Aktif | Cuaca Hujan)

jumlah_promo_hujan = (
    (df["Promo_Aktif"] == "Ya") &
    (df["Kondisi_Cuaca"] == "Hujan")
).sum()

P_A_given_C = jumlah_promo_hujan / C

print("\n2. Peluang Promo Aktif jika Cuaca Hujan")
print(f"P(A|C) = {jumlah_promo_hujan}/{C}")
print(f"P(A|C) = {P_A_given_C:.4f}")
print(f"Persentase = {P_A_given_C*100:.2f}%")


# ============================================================
# 5. DISTRIBUSI NORMAL
# ============================================================

print("\n" + "=" * 65)
print("BAGIAN 3: DISTRIBUSI NORMAL")
print("=" * 65)

# Menggunakan biaya operasional
biaya = df["Biaya_Operasional_Ribu"].dropna()

# Menghitung parameter distribusi normal
mu = biaya.mean()
sigma = biaya.std(ddof=1)

print("\nVariabel: Biaya_Operasional_Ribu")
print("Jumlah data:", len(biaya))
print(f"Mean (mu): {mu:.4f}")
print(f"Standar deviasi sampel: {sigma:.4f}")


# ------------------------------------------------------------
# 5A. UJI NORMALITAS SHAPIRO-WILK
# ------------------------------------------------------------

stat_shapiro, p_shapiro = stats.shapiro(biaya)

print("\nUJI NORMALITAS SHAPIRO-WILK")
print(f"Statistik W = {stat_shapiro:.4f}")
print(f"P-value = {p_shapiro:.6f}")

alpha = 0.05

if p_shapiro > alpha:
    print("Keputusan: Tidak menolak H0.")
    print("Interpretasi: Tidak terdapat bukti yang cukup")
    print("untuk menyatakan data menyimpang dari normalitas.")
else:
    print("Keputusan: Menolak H0.")
    print("Interpretasi: Data menunjukkan penyimpangan")
    print("yang signifikan dari distribusi normal.")


# ------------------------------------------------------------
# 5B. SKENARIO PELUANG NORMAL
# ------------------------------------------------------------

# Skenario 1: Biaya operasional kurang dari 1200
batas_1 = 1200

z1 = (batas_1 - mu) / sigma

peluang_1 = norm.cdf(
    batas_1,
    loc=mu,
    scale=sigma
)

print("\nSKENARIO 1")
print("P(X < 1200)")
print(f"Z-score = {z1:.4f}")
print(f"P(X < 1200) = {peluang_1:.4f}")
print(f"Persentase = {peluang_1*100:.2f}%")


# Skenario 2: Biaya operasional lebih dari 1600
batas_2 = 1600

z2 = (batas_2 - mu) / sigma

peluang_2 = norm.sf(
    batas_2,
    loc=mu,
    scale=sigma
)

print("\nSKENARIO 2")
print("P(X > 1600)")
print(f"Z-score = {z2:.4f}")
print(f"P(X > 1600) = {peluang_2:.4f}")
print(f"Persentase = {peluang_2*100:.2f}%")


# ------------------------------------------------------------
# 5C. VISUALISASI DISTRIBUSI NORMAL
# ------------------------------------------------------------

x = np.linspace(
    mu - 4*sigma,
    mu + 4*sigma,
    500
)

y = norm.pdf(
    x,
    loc=mu,
    scale=sigma
)

plt.figure(figsize=(10, 5))

plt.plot(
    x,
    y,
    linewidth=2,
    label="Kurva distribusi normal"
)

plt.axvline(
    batas_1,
    linestyle="--",
    label="Batas Rp1.200.000"
)

plt.axvline(
    batas_2,
    linestyle="--",
    label="Batas Rp1.600.000"
)

plt.axvline(
    mu,
    linestyle=":",
    label="Mean"
)

plt.title("Distribusi Normal Biaya Operasional UMKM Kopi")
plt.xlabel("Biaya Operasional (Ribu Rupiah)")
plt.ylabel("Kepadatan Probabilitas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "output/grafik/04_distribusi_normal.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. DISTRIBUSI BINOMIAL
# ============================================================

print("\n" + "=" * 65)
print("BAGIAN 4: DISTRIBUSI BINOMIAL")
print("=" * 65)

# Variabel biner:
# Sukses = Promo Aktif (Ya)
# Gagal = Promo Tidak Aktif (Tidak)

jumlah_sukses = (df["Promo_Aktif"] == "Ya").sum()

jumlah_total = len(df)

p = jumlah_sukses / jumlah_total

q = 1 - p

n = 20

k = 8

print("\nPARAMETER BINOMIAL")
print("Kejadian sukses: Promo Aktif")
print("Kejadian gagal: Promo Tidak Aktif")
print(f"Jumlah sukses = {jumlah_sukses}")
print(f"Jumlah observasi = {jumlah_total}")
print(f"Peluang sukses (p) = {p:.4f}")
print(f"Peluang gagal (q) = {q:.4f}")
print(f"Ukuran sampel (n) = {n}")


# ------------------------------------------------------------
# 6A. P(X = 8)
# ------------------------------------------------------------

peluang_tepat = binom.pmf(
    k,
    n,
    p
)

print("\nSKENARIO 1: TEPAT 8 KEBERHASILAN")

print(f"P(X = {k}) = {peluang_tepat:.6f}")

print(f"Persentase = {peluang_tepat*100:.2f}%")


# ------------------------------------------------------------
# 6B. P(X >= 8)
# ------------------------------------------------------------

peluang_minimal = binom.sf(
    k - 1,
    n,
    p
)

print("\nSKENARIO 2: MINIMAL 8 KEBERHASILAN")

print(f"P(X >= {k}) = {peluang_minimal:.6f}")

print(f"Persentase = {peluang_minimal*100:.2f}%")


# ------------------------------------------------------------
# 6C. TABEL DISTRIBUSI BINOMIAL
# ------------------------------------------------------------

nilai_x = np.arange(0, n + 1)

peluang_binomial = binom.pmf(
    nilai_x,
    n,
    p
)

tabel_binomial = pd.DataFrame({
    "Jumlah_Promo_Aktif": nilai_x,
    "Peluang": peluang_binomial,
    "Persentase": peluang_binomial * 100
})

print("\nTABEL DISTRIBUSI BINOMIAL")
print(tabel_binomial.round(5).to_string(index=False))


# ------------------------------------------------------------
# 6D. GRAFIK DISTRIBUSI BINOMIAL
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    nilai_x,
    peluang_binomial,
    edgecolor="black"
)

plt.title("Distribusi Binomial Promo Aktif")
plt.xlabel("Jumlah Promo Aktif (X)")
plt.ylabel("Peluang P(X)")
plt.xticks(nilai_x)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "output/grafik/05_distribusi_binomial.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. MENYIMPAN HASIL ANALISIS KE EXCEL
# ============================================================

print("\n" + "=" * 65)
print("MENYIMPAN HASIL ANALISIS")
print("=" * 65)

tabel_peluang = pd.DataFrame([
    ["P(Promo Aktif)", A, N, P_A],
    ["P(Pembayaran QRIS)", B, N, P_B],
    ["P(Cuaca Hujan)", C, N, P_C],
    ["P(Promo Aktif | QRIS)", jumlah_promo_qris, B, P_A_given_B],
    ["P(Promo Aktif | Hujan)", jumlah_promo_hujan, C, P_A_given_C]
], columns=[
    "Kejadian",
    "Frekuensi",
    "Jumlah_Pembanding",
    "Peluang"
])

tabel_normal = pd.DataFrame([
    ["P(X < 1200)", batas_1, z1, peluang_1],
    ["P(X > 1600)", batas_2, z2, peluang_2]
], columns=[
    "Skenario",
    "Batas_Nilai",
    "Z_Score",
    "Peluang"
])

hasil_uji_normal = pd.DataFrame([
    ["Biaya_Operasional_Ribu", len(biaya), mu, sigma,
     stat_shapiro, p_shapiro, alpha,
     "Tidak menolak H0" if p_shapiro > alpha else "Menolak H0"]
], columns=[
    "Variabel",
    "N",
    "Mean",
    "Standar_Deviasi",
    "Statistik_W",
    "P_Value",
    "Alpha",
    "Keputusan"
])

tabel_binomial_ringkasan = pd.DataFrame([
    ["P(X = 8)", n, p, k, peluang_tepat],
    ["P(X >= 8)", n, p, k, peluang_minimal]
], columns=[
    "Skenario",
    "n",
    "p",
    "k",
    "Peluang"
])

with pd.ExcelWriter(
    "output/hasil_analisis_minggu_2.xlsx"
) as writer:

    tabel_peluang.to_excel(
        writer,
        sheet_name="Peluang_Empiris",
        index=False
    )

    hasil_uji_normal.to_excel(
        writer,
        sheet_name="Uji_Normalitas",
        index=False
    )

    tabel_normal.to_excel(
        writer,
        sheet_name="Peluang_Normal",
        index=False
    )

    tabel_binomial_ringkasan.to_excel(
        writer,
        sheet_name="Binomial_Ringkasan",
        index=False
    )

    tabel_binomial.to_excel(
        writer,
        sheet_name="Distribusi_Binomial",
        index=False
    )


# ============================================================
# 8. RINGKASAN AKHIR
# ============================================================

print("\n" + "=" * 65)
print("ANALISIS MINGGU 2 BERHASIL DIJALANKAN!")
print("=" * 65)

print("\nPELUANG EMPIRIS")
print(f"P(Promo Aktif) = {P_A:.4f}")
print(f"P(Pembayaran QRIS) = {P_B:.4f}")
print(f"P(Cuaca Hujan) = {P_C:.4f}")

print("\nPELUANG NORMAL")
print(f"P(X < 1200) = {peluang_1:.4f}")
print(f"P(X > 1600) = {peluang_2:.4f}")

print("\nPELUANG BINOMIAL")
print(f"P(X = 8) = {peluang_tepat:.6f}")
print(f"P(X >= 8) = {peluang_minimal:.6f}")

print("\nHASIL EXCEL:")
print("output/hasil_analisis_minggu_2.xlsx")

print("\nGRAFIK:")
print("output/grafik/04_distribusi_normal.png")
print("output/grafik/05_distribusi_binomial.png")

print("\nSELURUH PROSES MINGGU 2 SELESAI!")