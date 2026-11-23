import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import stats
from scipy.stats import chi2_contingency

from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# 1. PERSIAPAN
# ============================================================

FILE = "dataset/Kelompok_36_Tema_D_UMKM_Kopi.csv"

ALPHA = 0.05

os.makedirs("output", exist_ok=True)
os.makedirs("output/grafik", exist_ok=True)

print("=" * 70)
print("PROJECT STATISTIKA DAN PROBABILITAS")
print("KELOMPOK 36 - TEMA D: UMKM KOPI")
print("MINGGU 3: INFERENSI STATISTIK")
print("=" * 70)

# ============================================================
# 2. MEMBACA DATASET
# ============================================================

df = pd.read_csv(FILE)

variabel_numerik = "Biaya_Operasional_Ribu"
variabel_biner = "Promo_Aktif"
variabel_kelompok = "Shift_Operasional"
variabel_kategori = "Metode_Bayar_Dominan"

df[variabel_numerik] = pd.to_numeric(
    df[variabel_numerik],
    errors="coerce"
)

print("\nINFORMASI DATASET")
print("Jumlah observasi:", len(df))
print("Jumlah variabel:", len(df.columns))

print("\nKategori promo:")
print(df[variabel_biner].value_counts(dropna=False))

print("\nKategori shift:")
print(df[variabel_kelompok].value_counts(dropna=False))

print("\nKategori pembayaran:")
print(df[variabel_kategori].value_counts(dropna=False))


# ============================================================
# 3. CONFIDENCE INTERVAL RATA-RATA POPULASI
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 1: CONFIDENCE INTERVAL RATA-RATA")
print("=" * 70)

data_mean = df[variabel_numerik].dropna()

n = len(data_mean)
mean = data_mean.mean()
std = data_mean.std(ddof=1)

# Menggunakan distribusi t karena standar deviasi populasi
# tidak diketahui

standard_error = std / np.sqrt(n)

t_critical = stats.t.ppf(
    1 - ALPHA / 2,
    df=n - 1
)

margin_error = t_critical * standard_error

ci_mean_lower = mean - margin_error
ci_mean_upper = mean + margin_error

print("\nVariabel:", variabel_numerik)
print("Jumlah data:", n)
print(f"Mean: {mean:.4f}")
print(f"Standar deviasi: {std:.4f}")
print(f"Standard error: {standard_error:.4f}")
print(f"t-kritis: {t_critical:.4f}")

print("\nCONFIDENCE INTERVAL 95%")
print(f"Batas bawah: {ci_mean_lower:.4f}")
print(f"Batas atas : {ci_mean_upper:.4f}")

print(
    f"CI 95% = ({ci_mean_lower:.4f}, "
    f"{ci_mean_upper:.4f})"
)


# ============================================================
# 4. CONFIDENCE INTERVAL PROPORSI POPULASI
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 2: CONFIDENCE INTERVAL PROPORSI")
print("=" * 70)

data_promo = df[variabel_biner].dropna().astype(str).str.strip()

jumlah_data_promo = len(data_promo)

jumlah_promo_aktif = (
    data_promo.str.lower() == "ya"
).sum()

proporsi = jumlah_promo_aktif / jumlah_data_promo

# Menggunakan metode Wilson untuk CI proporsi

ci_prop_lower, ci_prop_upper = proportion_confint(
    count=jumlah_promo_aktif,
    nobs=jumlah_data_promo,
    alpha=ALPHA,
    method="wilson"
)

print("\nVariabel:", variabel_biner)
print("Jumlah data:", jumlah_data_promo)
print("Jumlah promo aktif:", jumlah_promo_aktif)

print(f"Proporsi sampel: {proporsi:.4f}")
print(f"Persentase: {proporsi * 100:.2f}%")

print("\nCONFIDENCE INTERVAL PROPORSI 95%")
print(f"Batas bawah: {ci_prop_lower:.4f}")
print(f"Batas atas : {ci_prop_upper:.4f}")

print(
    f"CI 95% = ({ci_prop_lower:.4f}, "
    f"{ci_prop_upper:.4f})"
)


# ============================================================
# 5. TWO-SAMPLE INDEPENDENT T-TEST
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 3: TWO-SAMPLE INDEPENDENT T-TEST")
print("=" * 70)

# Membandingkan biaya operasional berdasarkan status promo

data_ttest = df[
    [variabel_numerik, variabel_biner]
].dropna().copy()

data_ttest[variabel_biner] = (
    data_ttest[variabel_biner]
    .astype(str)
    .str.strip()
)

kelompok_ya = data_ttest.loc[
    data_ttest[variabel_biner].str.lower() == "ya",
    variabel_numerik
]

kelompok_tidak = data_ttest.loc[
    data_ttest[variabel_biner].str.lower() == "tidak",
    variabel_numerik
]

if len(kelompok_ya) < 2 or len(kelompok_tidak) < 2:
    raise ValueError(
        "Data untuk kedua kelompok t-test tidak mencukupi."
    )

# H0: Rata-rata biaya operasional kedua kelompok sama.
# H1: Rata-rata biaya operasional kedua kelompok berbeda.

t_stat, p_ttest = stats.ttest_ind(
    kelompok_ya,
    kelompok_tidak,
    equal_var=False
)

print("\nHIPOTESIS")
print("H0: Rata-rata biaya operasional promo aktif")
print("    dan tidak aktif adalah sama.")
print("H1: Rata-rata biaya operasional kedua kelompok berbeda.")

print("\nSTATISTIK DESKRIPTIF KELOMPOK")
print("Promo aktif")
print("Jumlah data:", len(kelompok_ya))
print(f"Mean: {kelompok_ya.mean():.4f}")

print("\nPromo tidak aktif")
print("Jumlah data:", len(kelompok_tidak))
print(f"Mean: {kelompok_tidak.mean():.4f}")

print("\nHASIL INDEPENDENT T-TEST")
print(f"Statistik t: {t_stat:.4f}")
print(f"P-value: {p_ttest:.6f}")

if p_ttest < ALPHA:
    keputusan_ttest = "Tolak H0"
    interpretasi_ttest = (
        "Terdapat perbedaan rata-rata biaya operasional "
        "yang signifikan antara kelompok promo aktif "
        "dan tidak aktif."
    )
else:
    keputusan_ttest = "Tidak menolak H0"
    interpretasi_ttest = (
        "Belum terdapat bukti statistik yang cukup "
        "untuk menyatakan adanya perbedaan rata-rata "
        "biaya operasional antara kedua kelompok."
    )

print("Keputusan:", keputusan_ttest)
print("Interpretasi:", interpretasi_ttest)


# ============================================================
# 6. ONE-WAY ANOVA
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 4: ONE-WAY ANOVA")
print("=" * 70)

# Membandingkan rata-rata biaya operasional
# berdasarkan shift operasional

data_anova = df[
    [variabel_numerik, variabel_kelompok]
].dropna().copy()

data_anova[variabel_kelompok] = (
    data_anova[variabel_kelompok]
    .astype(str)
    .str.strip()
)

kelompok_anova = []

for nama, grup in data_anova.groupby(variabel_kelompok):
    kelompok_anova.append(grup[variabel_numerik].values)

    print(f"\nKelompok: {nama}")
    print("Jumlah data:", len(grup))
    print(f"Mean: {grup[variabel_numerik].mean():.4f}")

if len(kelompok_anova) < 2:
    raise ValueError(
        "ANOVA memerlukan minimal dua kelompok."
    )

# H0: Seluruh rata-rata biaya operasional antar shift sama.
# H1: Minimal ada satu rata-rata yang berbeda.

f_stat, p_anova = stats.f_oneway(*kelompok_anova)

print("\nHIPOTESIS")
print("H0: Rata-rata biaya operasional seluruh shift sama.")
print("H1: Minimal terdapat satu rata-rata shift berbeda.")

print("\nHASIL ONE-WAY ANOVA")
print(f"Statistik F: {f_stat:.4f}")
print(f"P-value: {p_anova:.6f}")

hasil_tukey = None

if p_anova < ALPHA:

    keputusan_anova = "Tolak H0"

    interpretasi_anova = (
        "Terdapat perbedaan rata-rata biaya operasional "
        "yang signifikan pada minimal dua kelompok shift."
    )

    print("\nANOVA SIGNIFIKAN")
    print("Melanjutkan pengujian Post-Hoc Tukey.")

    hasil_tukey = pairwise_tukeyhsd(
        endog=data_anova[variabel_numerik],
        groups=data_anova[variabel_kelompok],
        alpha=ALPHA
    )

    print("\nHASIL POST-HOC TUKEY")
    print(hasil_tukey)

else:

    keputusan_anova = "Tidak menolak H0"

    interpretasi_anova = (
        "Belum terdapat bukti statistik yang cukup "
        "untuk menyatakan adanya perbedaan rata-rata "
        "biaya operasional antar shift."
    )

    print("\nANOVA TIDAK SIGNIFIKAN")
    print("Uji Post-Hoc Tukey tidak dilanjutkan.")

print("\nKeputusan:", keputusan_anova)
print("Interpretasi:", interpretasi_anova)


# ============================================================
# 7. CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 5: CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 70)

# Menguji hubungan antara status promo
# dan metode pembayaran dominan

data_chi = df[
    [variabel_biner, variabel_kategori]
].dropna().copy()

data_chi[variabel_biner] = (
    data_chi[variabel_biner]
    .astype(str)
    .str.strip()
)

data_chi[variabel_kategori] = (
    data_chi[variabel_kategori]
    .astype(str)
    .str.strip()
)

tabel_observasi = pd.crosstab(
    data_chi[variabel_biner],
    data_chi[variabel_kategori]
)

chi2_stat, p_chi, df_chi, expected = chi2_contingency(
    tabel_observasi
)

tabel_ekspektasi = pd.DataFrame(
    expected,
    index=tabel_observasi.index,
    columns=tabel_observasi.columns
)

print("\nHIPOTESIS")
print("H0: Tidak terdapat hubungan antara status promo")
print("    dan metode pembayaran dominan.")
print("H1: Terdapat hubungan antara status promo")
print("    dan metode pembayaran dominan.")

print("\nTABEL FREKUENSI OBSERVASI")
print(tabel_observasi)

print("\nTABEL FREKUENSI EKSPEKTASI")
print(tabel_ekspektasi.round(2))

print("\nHASIL CHI-SQUARE")
print(f"Statistik Chi-Square: {chi2_stat:.4f}")
print(f"Derajat kebebasan: {df_chi}")
print(f"P-value: {p_chi:.6f}")

# Menghitung Cramer's V sebagai ukuran kekuatan hubungan

n_chi = tabel_observasi.to_numpy().sum()

min_dimensi = min(tabel_observasi.shape) - 1

if min_dimensi > 0 and n_chi > 0:
    cramers_v = np.sqrt(
        chi2_stat / (n_chi * min_dimensi)
    )
else:
    cramers_v = np.nan

print(f"Cramer's V: {cramers_v:.4f}")

if p_chi < ALPHA:

    keputusan_chi = "Tolak H0"

    interpretasi_chi = (
        "Terdapat hubungan yang signifikan secara statistik "
        "antara status promo dan metode pembayaran dominan."
    )

else:

    keputusan_chi = "Tidak menolak H0"

    interpretasi_chi = (
        "Belum terdapat bukti statistik yang cukup "
        "untuk menyatakan adanya hubungan antara "
        "status promo dan metode pembayaran dominan."
    )

print("Keputusan:", keputusan_chi)
print("Interpretasi:", interpretasi_chi)

# Memeriksa frekuensi ekspektasi
jumlah_expected_kecil = (expected < 5).sum()

print("\nPEMERIKSAAN ASUMSI CHI-SQUARE")
print(
    "Jumlah sel dengan frekuensi ekspektasi < 5:",
    jumlah_expected_kecil
)

if jumlah_expected_kecil > 0:
    print(
        "PERHATIAN: Ada frekuensi ekspektasi di bawah 5. "
        "Asumsi Chi-Square perlu diperiksa lebih lanjut."
    )
else:
    print(
        "Seluruh frekuensi ekspektasi minimal 5."
    )


# ============================================================
# 8. VISUALISASI
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 6: VISUALISASI")
print("=" * 70)

# Grafik 1: CI rata-rata biaya operasional

plt.figure(figsize=(8, 5))

plt.errorbar(
    x=[mean],
    y=[0],
    xerr=[[
        mean - ci_mean_lower
    ], [
        ci_mean_upper - mean
    ]],
    fmt="o",
    capsize=8
)

plt.yticks([])
plt.xlabel("Biaya Operasional (Ribu Rupiah)")
plt.title("Confidence Interval 95% Rata-rata Biaya Operasional")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "output/grafik/06_ci_mean.png",
    dpi=300
)

plt.close()


# Grafik 2: Rata-rata biaya berdasarkan promo

rata_promo = data_ttest.groupby(
    variabel_biner
)[variabel_numerik].mean()

plt.figure(figsize=(8, 5))

rata_promo.plot(kind="bar")

plt.title("Rata-rata Biaya Operasional Berdasarkan Promo")
plt.xlabel("Status Promo")
plt.ylabel("Biaya Operasional (Ribu Rupiah)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "output/grafik/07_ttest_promo.png",
    dpi=300
)

plt.close()


# Grafik 3: Rata-rata biaya berdasarkan shift

rata_shift = data_anova.groupby(
    variabel_kelompok
)[variabel_numerik].mean()

plt.figure(figsize=(8, 5))

rata_shift.plot(kind="bar")

plt.title("Rata-rata Biaya Operasional Berdasarkan Shift")
plt.xlabel("Shift Operasional")
plt.ylabel("Biaya Operasional (Ribu Rupiah)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "output/grafik/08_anova_shift.png",
    dpi=300
)

plt.close()


# Grafik 4: Heatmap frekuensi observasi Chi-Square

plt.figure(figsize=(8, 5))

plt.imshow(
    tabel_observasi.values,
    aspect="auto"
)

plt.xticks(
    range(len(tabel_observasi.columns)),
    tabel_observasi.columns,
    rotation=30
)

plt.yticks(
    range(len(tabel_observasi.index)),
    tabel_observasi.index
)

plt.colorbar(label="Frekuensi Observasi")

plt.title("Kontingensi Promo dan Metode Pembayaran")

plt.xlabel("Metode Pembayaran")
plt.ylabel("Status Promo")

plt.tight_layout()

plt.savefig(
    "output/grafik/09_chisquare.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. MENYIMPAN HASIL ANALISIS KE EXCEL
# ============================================================

print("\n" + "=" * 70)
print("BAGIAN 7: MENYIMPAN HASIL ANALISIS")
print("=" * 70)

ringkasan_ci = pd.DataFrame([
    [
        "CI Mean Biaya Operasional",
        n,
        mean,
        ci_mean_lower,
        ci_mean_upper
    ],
    [
        "CI Proporsi Promo Aktif",
        jumlah_data_promo,
        proporsi,
        ci_prop_lower,
        ci_prop_upper
    ]
], columns=[
    "Analisis",
    "Jumlah_Data",
    "Estimasi",
    "Batas_Bawah_95",
    "Batas_Atas_95"
])

ringkasan_uji = pd.DataFrame([
    [
        "Independent t-Test",
        t_stat,
        np.nan,
        p_ttest,
        keputusan_ttest,
        interpretasi_ttest
    ],
    [
        "One-Way ANOVA",
        f_stat,
        np.nan,
        p_anova,
        keputusan_anova,
        interpretasi_anova
    ],
    [
        "Chi-Square",
        chi2_stat,
        df_chi,
        p_chi,
        keputusan_chi,
        interpretasi_chi
    ]
], columns=[
    "Nama_Uji",
    "Statistik_Uji",
    "Derajat_Kebebasan",
    "P_Value",
    "Keputusan",
    "Interpretasi"
])

ringkasan_kelompok = pd.DataFrame([
    {
        "Kelompok": "Promo Aktif",
        "Jumlah": len(kelompok_ya),
        "Mean": kelompok_ya.mean(),
        "Standar_Deviasi": kelompok_ya.std()
    },
    {
        "Kelompok": "Promo Tidak Aktif",
        "Jumlah": len(kelompok_tidak),
        "Mean": kelompok_tidak.mean(),
        "Standar_Deviasi": kelompok_tidak.std()
    }
])

with pd.ExcelWriter(
    "output/hasil_analisis_minggu_3.xlsx"
) as writer:

    ringkasan_ci.to_excel(
        writer,
        sheet_name="Confidence_Interval",
        index=False
    )

    ringkasan_uji.to_excel(
        writer,
        sheet_name="Ringkasan_Uji",
        index=False
    )

    ringkasan_kelompok.to_excel(
        writer,
        sheet_name="Deskriptif_TTest",
        index=False
    )

    tabel_observasi.to_excel(
        writer,
        sheet_name="Chi_Observed"
    )

    tabel_ekspektasi.to_excel(
        writer,
        sheet_name="Chi_Expected"
    )

    rata_shift.to_frame(
        name="Rata_Rata_Biaya"
    ).to_excel(
        writer,
        sheet_name="Deskriptif_ANOVA"
    )

    if hasil_tukey is not None:

        tukey_df = pd.DataFrame(
            data=hasil_tukey._results_table.data[1:],
            columns=hasil_tukey._results_table.data[0]
        )

        tukey_df.to_excel(
            writer,
            sheet_name="Post_Hoc_Tukey",
            index=False
        )


# ============================================================
# 10. RINGKASAN AKHIR
# ============================================================

print("\n" + "=" * 70)
print("ANALISIS MINGGU 3 SELESAI!")
print("=" * 70)

print("\nCONFIDENCE INTERVAL")
print(
    f"CI Mean: ({ci_mean_lower:.4f}, "
    f"{ci_mean_upper:.4f})"
)

print(
    f"CI Proporsi: ({ci_prop_lower:.4f}, "
    f"{ci_prop_upper:.4f})"
)

print("\nHASIL UJI HIPOTESIS")

print("Independent t-Test:", keputusan_ttest)
print("One-Way ANOVA:", keputusan_anova)
print("Chi-Square:", keputusan_chi)

print("\nFILE HASIL:")
print("output/hasil_analisis_minggu_3.xlsx")

print("\nGRAFIK:")
print("output/grafik/06_ci_mean.png")
print("output/grafik/07_ttest_promo.png")
print("output/grafik/08_anova_shift.png")
print("output/grafik/09_chisquare.png")

print("\nSELURUH PROSES MINGGU 3 SELESAI!")
print("=" * 70)
