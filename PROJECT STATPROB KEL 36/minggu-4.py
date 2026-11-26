from pathlib import Path
from itertools import combinations
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. KONFIGURASI FOLDER DAN DATASET
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent

# File hasil pengolahan Progres 1 yang menjadi sumber data Progres 4
FILE_DATA = BASE_DIR / "output" / "hasil_analisis_minggu_1.xlsx"
SHEET_DATA = "Data_Analisis"

# Semua hasil Progres 4 langsung disimpan di folder output yang sama
FOLDER_OUTPUT = BASE_DIR / "output"
FOLDER_GRAFIK = FOLDER_OUTPUT / "grafik"

FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
FOLDER_GRAFIK.mkdir(parents=True, exist_ok=True)

FILE_HASIL = FOLDER_OUTPUT / "hasil_analisis_minggu_4.xlsx"
FILE_RINGKASAN = FOLDER_OUTPUT / "ringkasan_minggu_4.txt"

# =====================================================================
# 2. NAMA VARIABEL
# =====================================================================
# Sesuaikan nama di bawah HANYA jika nama kolom pada sheet Excel berbeda.
# Berdasarkan analisis sebelumnya:
# X1 = Jumlah_Pengunjung
# X2 = Suhu_Rata2
# X3 = Biaya_Operasional_Ribu
# Y  = Omzet_Harian_Ribu

VARIABEL_X = [
    "Jumlah_Pengunjung",
    "Suhu_Rata2",
    "Biaya_Operasional_Ribu",
]
VARIABEL_Y = "Omzet_Harian_Ribu"
VARIABEL_ANALISIS = VARIABEL_X + [VARIABEL_Y]

sns.set_theme(style="whitegrid")


def nomor_grafik_berikutnya():
    """
    Mencari nomor grafik terbesar yang sudah ada di folder grafik,
    lalu mengembalikan nomor berikutnya agar penomoran Progres 4
    melanjutkan grafik dari Progres sebelumnya.

    Mendukung nama seperti:
    01_nama_grafik.png
    grafik_01_nama.png
    01-grafik.png
    minggu4_01_nama.png
    """
    nomor_terbesar = 0

    pola_nama = re.compile(
        r"^(?:(?:grafik|minggu\d+)[_-]?)?(\d+)(?:[_-]|\.)",
        re.IGNORECASE
    )
    pola_minggu = re.compile(
        r"^minggu\d+[_-](\d+)(?:[_-]|\.)",
        re.IGNORECASE
    )

    for file_gambar in FOLDER_GRAFIK.iterdir():
        if not file_gambar.is_file():
            continue

        cocok = pola_nama.match(file_gambar.name)
        if not cocok:
            cocok = pola_minggu.match(file_gambar.name)

        if cocok:
            nomor_terbesar = max(nomor_terbesar, int(cocok.group(1)))

    return nomor_terbesar + 1


NOMOR_GRAFIK = nomor_grafik_berikutnya()


def simpan_gambar(nama_deskriptif):
    """Menyimpan grafik dengan nomor urut lanjutan, lalu menutup figure."""
    global NOMOR_GRAFIK

    plt.tight_layout()
    nama_file = f"{NOMOR_GRAFIK:02d}_{nama_deskriptif}.png"
    plt.savefig(FOLDER_GRAFIK / nama_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grafik tersimpan: {nama_file}")
    NOMOR_GRAFIK += 1


# =====================================================================
# 3. MEMBACA DAN MEMERIKSA DATA
# =====================================================================

print("=" * 72)
print("ANALISIS PROGRES 4 - STATPROB KELOMPOK 36")
print("KORELASI, REGRESI, DAN REKOMENDASI UMKM KOPI")
print("=" * 72)

print("\n[1] MEMERIKSA DATASET")
print(f"Lokasi dataset: {FILE_DATA}")

if not FILE_DATA.exists():
    raise FileNotFoundError(
        "\nFile dataset tidak ditemukan.\n"
        f"Program mencari file di:\n{FILE_DATA}\n\n"
        "Pastikan hasil_analisis_minggu_1.xlsx berada di folder output."
    )

xls = pd.ExcelFile(FILE_DATA)
print("Sheet yang tersedia:", xls.sheet_names)

if SHEET_DATA not in xls.sheet_names:
    raise ValueError(
        f"Sheet '{SHEET_DATA}' tidak ditemukan. "
        f"Sheet yang tersedia: {xls.sheet_names}. "
        "Periksa nama sheet dan ubah SHEET_DATA di bagian konfigurasi."
    )

df_asli = pd.read_excel(FILE_DATA, sheet_name=SHEET_DATA)

kolom_tidak_ada = [
    kolom for kolom in VARIABEL_ANALISIS if kolom not in df_asli.columns
]
if kolom_tidak_ada:
    raise ValueError(
        "Kolom berikut tidak ditemukan pada dataset: "
        + ", ".join(kolom_tidak_ada)
        + "\nKolom yang tersedia: "
        + ", ".join(map(str, df_asli.columns))
        + "\nSesuaikan nama variabel pada bagian konfigurasi kode."
    )

data = df_asli[VARIABEL_ANALISIS].copy()
for kolom in VARIABEL_ANALISIS:
    data[kolom] = pd.to_numeric(data[kolom], errors="coerce")

jumlah_data_awal = len(data)
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
jumlah_data_dihapus = jumlah_data_awal - len(data)

if len(data) <= len(VARIABEL_X) + 1:
    raise ValueError("Observasi tidak cukup untuk membangun regresi linear berganda.")

kolom_konstan = [kolom for kolom in VARIABEL_ANALISIS if data[kolom].nunique() <= 1]
if kolom_konstan:
    raise ValueError(
        "Variabel berikut tidak memiliki variasi data: "
        + ", ".join(kolom_konstan)
    )

print(f"Jumlah observasi awal       : {jumlah_data_awal}")
print(f"Jumlah observasi digunakan  : {len(data)}")
print(f"Baris kosong/tidak valid    : {jumlah_data_dihapus}")
print("Variabel analisis           :", ", ".join(VARIABEL_ANALISIS))

# =====================================================================
# 4A. KORELASI PEARSON SELURUH VARIABEL ANALISIS
# =====================================================================

print("\n[2] KORELASI PEARSON")

korelasi = data[VARIABEL_ANALISIS].corr(method="pearson")
print("\nMatriks Korelasi Pearson:")
print(korelasi.round(4).to_string())

# P-value korelasi untuk setiap pasangan variabel
p_value_korelasi = pd.DataFrame(
    np.nan, index=VARIABEL_ANALISIS, columns=VARIABEL_ANALISIS
)
for variabel in VARIABEL_ANALISIS:
    p_value_korelasi.loc[variabel, variabel] = 0.0

for x, y in combinations(VARIABEL_ANALISIS, 2):
    r, p = stats.pearsonr(data[x], data[y])
    p_value_korelasi.loc[x, y] = p
    p_value_korelasi.loc[y, x] = p

# Heatmap matriks korelasi
plt.figure(figsize=(10, 7))
sns.heatmap(
    korelasi,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
)
plt.title("Matriks Korelasi Pearson Antarvariabel Numerik")
simpan_gambar("matriks_korelasi")

# =====================================================================
# 4A. SCATTER PLOT DENGAN GARIS TREN UNTUK SETIAP PASANGAN
# =====================================================================

print("\nMembuat scatter plot dan garis tren...")
hasil_scatter = []

for nomor, (x, y) in enumerate(combinations(VARIABEL_ANALISIS, 2), start=1):
    r, p = stats.pearsonr(data[x], data[y])

    hasil_scatter.append({
        "Variabel_X": x,
        "Variabel_Y": y,
        "Korelasi_Pearson": r,
        "P_Value": p,
    })

    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=data,
        x=x,
        y=y,
        ci=95,
        scatter_kws={"alpha": 0.65},
        line_kws={"linewidth": 2},
    )
    plt.title(f"Scatter Plot {x} terhadap {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    simpan_gambar(f"scatter_{nomor:02d}")

hasil_scatter = pd.DataFrame(hasil_scatter)
print(f"{len(hasil_scatter)} scatter plot berhasil dibuat.")

# =====================================================================
# 4B. REGRESI LINEAR BERGANDA
# =====================================================================

print("\n[3] REGRESI LINEAR BERGANDA")

X = sm.add_constant(data[VARIABEL_X], has_constant="add")
Y = data[VARIABEL_Y]

model = sm.OLS(Y, X).fit()

print("\nRingkasan model:")
print(model.summary())

koef = model.params
persamaan = (
    f"Ŷ = {koef['const']:.6f}"
    f" + ({koef[VARIABEL_X[0]]:.6f} × X1)"
    f" + ({koef[VARIABEL_X[1]]:.6f} × X2)"
    f" + ({koef[VARIABEL_X[2]]:.6f} × X3)"
)

print("\nPersamaan regresi:")
print(persamaan)
print("\nKeterangan:")
print("X1 =", VARIABEL_X[0])
print("X2 =", VARIABEL_X[1])
print("X3 =", VARIABEL_X[2])
print("Y  =", VARIABEL_Y)

# =====================================================================
# 4C. R-SQUARED, ADJUSTED R-SQUARED, UJI-F, DAN UJI-t
# =====================================================================

print("\n[4] EVALUASI MODEL")

r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj
f_statistic = model.fvalue
p_value_f = model.f_pvalue

if p_value_f < 0.05:
    keputusan_f = "Signifikan secara simultan pada taraf 5%"
else:
    keputusan_f = "Tidak signifikan secara simultan pada taraf 5%"

evaluasi_model = pd.DataFrame({
    "Indikator": [
        "Jumlah observasi",
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "P-value Uji-F",
    ],
    "Nilai": [
        len(data),
        r_squared,
        adjusted_r_squared,
        f_statistic,
        p_value_f,
    ],
})

print(f"R-squared          : {r_squared:.6f}")
print(f"Adjusted R-squared : {adjusted_r_squared:.6f}")
print(f"F-statistic        : {f_statistic:.6f}")
print(f"P-value Uji-F      : {p_value_f:.10g}")
print("Keputusan Uji-F    :", keputusan_f)

# Tabel Uji-t; konstanta disertakan sebagai informasi, bukan prediktor
ci = model.conf_int()
hasil_uji_t = pd.DataFrame({
    "Koefisien": model.params,
    "Std_Error": model.bse,
    "t_statistic": model.tvalues,
    "P_Value": model.pvalues,
    "CI_95_Bawah": ci[0],
    "CI_95_Atas": ci[1],
})
hasil_uji_t.index.name = "Variabel"

print("\nHasil Uji-t:")
print(hasil_uji_t.round(6).to_string())

# =====================================================================
# 4D. PEMERIKSAAN ASUMSI RESIDUAL
# =====================================================================

print("\n[5] PEMERIKSAAN ASUMSI RESIDUAL")

fitted = model.fittedvalues
residual = model.resid

# Histogram residual
plt.figure(figsize=(8, 6))
plt.hist(residual, bins=25, density=True, alpha=0.65, edgecolor="black")
x_grid = np.linspace(residual.min(), residual.max(), 300)
if residual.std(ddof=1) > 0:
    plt.plot(x_grid, stats.gaussian_kde(residual)(x_grid), linewidth=2)
plt.axvline(0, linestyle="--", linewidth=1.5)
plt.title("Histogram Residual")
plt.xlabel("Residual")
plt.ylabel("Kepadatan")
simpan_gambar("histogram_residual")

# Q-Q plot residual
plt.figure(figsize=(8, 6))
sm.qqplot(residual, line="45", fit=True, ax=plt.gca())
plt.title("Q-Q Plot Residual")
simpan_gambar("qqplot_residual")

# Uji Shapiro-Wilk: untuk sampel > 5000, gunakan sampel acak maksimum 5000
if len(residual) > 5000:
    residual_uji = residual.sample(5000, random_state=42)
else:
    residual_uji = residual

shapiro_stat, shapiro_p = stats.shapiro(residual_uji)

if shapiro_p > 0.05:
    keputusan_normalitas = "Tidak terdapat bukti yang cukup untuk menolak normalitas residual"
else:
    keputusan_normalitas = "Terdapat indikasi residual tidak berdistribusi normal"

print("\nUji Shapiro-Wilk:")
print(f"Statistik : {shapiro_stat:.6f}")
print(f"P-value   : {shapiro_p:.10g}")
print("Kesimpulan:", keputusan_normalitas)

# Fitted values vs residuals
plt.figure(figsize=(8, 6))
plt.scatter(fitted, residual, alpha=0.65)
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.title("Fitted Values vs Residuals")
plt.xlabel("Fitted Values")
plt.ylabel("Residual")
simpan_gambar("fitted_vs_residual")

# Uji Breusch-Pagan sebagai pemeriksaan tambahan homoskedastisitas
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residual, X)

if bp_lm_p > 0.05:
    keputusan_bp = "Tidak terdapat bukti yang cukup untuk menyatakan heteroskedastisitas"
else:
    keputusan_bp = "Terdapat indikasi heteroskedastisitas"

print("\nUji Breusch-Pagan:")
print(f"LM Statistic : {bp_lm:.6f}")
print(f"P-value LM   : {bp_lm_p:.10g}")
print("Kesimpulan   :", keputusan_bp)

diagnostik_residual = pd.DataFrame({
    "Pengujian": ["Shapiro-Wilk", "Breusch-Pagan"],
    "Statistik": [shapiro_stat, bp_lm],
    "P_Value": [shapiro_p, bp_lm_p],
    "Kesimpulan": [keputusan_normalitas, keputusan_bp],
})

# =====================================================================
# PEMERIKSAAN TAMBAHAN: MULTIKOLINEARITAS (VIF)
# =====================================================================

print("\n[6] PEMERIKSAAN MULTIKOLINEARITAS (VIF)")

hasil_vif = pd.DataFrame({
    "Variabel": VARIABEL_X,
    "VIF": [
        variance_inflation_factor(X.values, i + 1)
        for i in range(len(VARIABEL_X))
    ],
})

print(hasil_vif.round(4).to_string(index=False))

# =====================================================================
# 4E. REKOMENDASI KEPUTUSAN BERDASARKAN HASIL STATISTIK
# =====================================================================

print("\n[7] REKOMENDASI KEPUTUSAN UMKM KOPI")

rekomendasi = []

# Jumlah pengunjung
p_pengunjung = model.pvalues[VARIABEL_X[0]]
b_pengunjung = model.params[VARIABEL_X[0]]

if p_pengunjung < 0.05 and b_pengunjung > 0:
    rekomendasi.append(
        "1. Memprioritaskan evaluasi dan penguatan promosi serta pelayanan "
        "untuk meningkatkan jumlah pengunjung. Dalam model, jumlah pengunjung "
        "memiliki koefisien positif dan signifikan terhadap omzet, dengan "
        "prediktor lain dianggap tetap."
    )
elif p_pengunjung < 0.05 and b_pengunjung < 0:
    rekomendasi.append(
        "1. Meninjau lebih lanjut hubungan jumlah pengunjung dengan omzet. "
        "Koefisien parsial yang negatif perlu ditafsirkan hati-hati, terutama "
        "karena kemungkinan keterkaitan antar-prediktor."
    )
else:
    rekomendasi.append(
        "1. Tetap memantau jumlah pengunjung dan omzet, tetapi jangan "
        "menjadikan jumlah pengunjung sebagai satu-satunya dasar target omzet "
        "karena pengaruh parsialnya belum signifikan pada taraf 5%."
    )

# Suhu
p_suhu = model.pvalues[VARIABEL_X[1]]
b_suhu = model.params[VARIABEL_X[1]]

if p_suhu < 0.05:
    arah_suhu = "positif" if b_suhu > 0 else "negatif"
    rekomendasi.append(
        f"2. Mempertimbangkan variabel suhu dalam evaluasi operasional karena "
        f"hubungan parsialnya dengan omzet terdeteksi signifikan dan arahnya "
        f"{arah_suhu}. Temuan ini bukan bukti sebab-akibat."
    )
else:
    rekomendasi.append(
        "2. Tidak menjadikan suhu sebagai dasar utama penetapan target omzet "
        "karena pengaruh parsialnya belum signifikan secara statistik."
    )

# Biaya operasional
p_biaya = model.pvalues[VARIABEL_X[2]]
b_biaya = model.params[VARIABEL_X[2]]

if p_biaya < 0.05 and b_biaya < 0:
    rekomendasi.append(
        "3. Meninjau pos-pos biaya operasional dan mengidentifikasi "
        "pengeluaran yang dapat diefisienkan, karena koefisien parsial biaya "
        "operasional negatif dan signifikan."
    )
elif p_biaya < 0.05 and b_biaya > 0:
    rekomendasi.append(
        "3. Mengukur produktivitas setiap pos biaya operasional. Koefisien "
        "positif tidak berarti biaya perlu dinaikkan tanpa analisis manfaat "
        "dan efisiensi."
    )
else:
    rekomendasi.append(
        "3. Memperkuat pencatatan biaya berdasarkan pos pengeluaran dan "
        "mengevaluasi efisiensinya. Pengaruh parsial total biaya operasional "
        "belum signifikan pada taraf 5%."
    )

# Multikolinearitas
if hasil_vif["VIF"].max() > 10:
    rekomendasi.append(
        "4. Menafsirkan koefisien parsial dengan hati-hati dan mengevaluasi "
        "keterkaitan antar-prediktor. Nilai VIF di atas 10 menunjukkan "
        "indikasi multikolinearitas kuat; pertimbangkan pengumpulan data "
        "tambahan atau peninjauan spesifikasi model."
    )
elif hasil_vif["VIF"].max() > 5:
    rekomendasi.append(
        "4. Memantau keterkaitan antar-prediktor karena terdapat indikasi "
        "multikolinearitas tingkat sedang berdasarkan VIF."
    )
else:
    rekomendasi.append(
        "4. Mempertahankan pencatatan variabel prediktor secara konsisten "
        "dan tetap memantau multikolinearitas pada pembaruan data berikutnya."
    )

for item in rekomendasi:
    print("\n" + item)

# =====================================================================
# 8. MENYIAPKAN DATA PREDIKSI DAN RESIDUAL
# =====================================================================

hasil_prediksi = data.copy()
hasil_prediksi["Omzet_Prediksi"] = fitted
hasil_prediksi["Residual"] = residual

# =====================================================================
# 9. MENYIMPAN SEMUA HASIL KE SATU FILE EXCEL
# =====================================================================

print("\n[8] MENYIMPAN HASIL ANALISIS")

with pd.ExcelWriter(FILE_HASIL, engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="Data_Analisis", index=False)
    korelasi.to_excel(writer, sheet_name="Korelasi_Pearson")
    p_value_korelasi.to_excel(writer, sheet_name="P_Value_Korelasi")
    hasil_scatter.to_excel(writer, sheet_name="Hasil_Scatter", index=False)
    evaluasi_model.to_excel(writer, sheet_name="Evaluasi_Model", index=False)
    hasil_uji_t.to_excel(writer, sheet_name="Uji_T")
    diagnostik_residual.to_excel(
        writer, sheet_name="Diagnostik_Residual", index=False
    )
    hasil_vif.to_excel(writer, sheet_name="Multikolinearitas", index=False)
    hasil_prediksi.to_excel(writer, sheet_name="Prediksi_Residual", index=False)

# Simpan ringkasan teks
with open(FILE_RINGKASAN, "w", encoding="utf-8") as file:
    file.write("STATPROB KELOMPOK 36 - PROGRES 4\n")
    file.write("=" * 72 + "\n\n")

    file.write("SUMBER DATA\n")
    file.write(f"File: {FILE_DATA.name}\n")
    file.write(f"Sheet: {SHEET_DATA}\n")
    file.write(f"Jumlah observasi digunakan: {len(data)}\n\n")

    file.write("PERSAMAAN REGRESI\n")
    file.write(persamaan + "\n\n")

    file.write("EVALUASI MODEL\n")
    file.write(evaluasi_model.to_string(index=False) + "\n\n")

    file.write("HASIL UJI-t\n")
    file.write(hasil_uji_t.to_string() + "\n\n")

    file.write("DIAGNOSTIK RESIDUAL\n")
    file.write(diagnostik_residual.to_string(index=False) + "\n\n")

    file.write("MULTIKOLINEARITAS (VIF)\n")
    file.write(hasil_vif.to_string(index=False) + "\n\n")

    file.write("REKOMENDASI KEPUTUSAN\n")
    for item in rekomendasi:
        file.write(item + "\n\n")

# =====================================================================
# 10. RINGKASAN AKHIR
# =====================================================================

print("\n" + "=" * 72)
print("ANALISIS PROGRES 4 SELESAI")
print("=" * 72)
print(f"File Excel hasil : {FILE_HASIL}")
print(f"Ringkasan teks   : {FILE_RINGKASAN}")
print(f"Folder grafik    : {FOLDER_GRAFIK}")
print("\nGrafik yang dibuat:")
for file_grafik in sorted(FOLDER_GRAFIK.glob("*.png")):
    print(" -", file_grafik.name)

print("\nSemua hasil Progres 4 tersimpan langsung di folder output.")