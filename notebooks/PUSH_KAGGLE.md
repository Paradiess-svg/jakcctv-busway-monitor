# Update notebook Kaggle otomatis dari GitHub

Notebook publik: <https://www.kaggle.com/code/izdiharwibowo/dasbor-tj-vs-non-tj>

Sumber kebenaran notebook = `notebooks/dasbor_kaggle.ipynb` di repo ini.
Setiap push ke `main` yang mengubah file itu (atau `kernel-metadata.json`) akan
otomatis membuat versi baru di Kaggle lewat GitHub Actions. Tidak perlu upload
manual, tidak perlu install apa pun di laptop.

Perintah yang dipakai workflow:

```bash
kaggle kernels push -p notebooks --no-run
```

`--no-run` = setara tombol **Quick Save** di web Kaggle: versi baru dibuat tanpa
menjalankan sel. Ini penting karena sel 6 berisi `while True` (monitor status),
jadi kalau dijalankan otomatis (Run All) sesinya akan menggantung sampai batas
waktu. Kode dasbor sendiri selalu di-clone dari GitHub oleh sel 1, jadi
perubahan kode (`src/`, `dashboard/`, `config/`) **tidak** butuh push notebook.

## Sekali saja: bikin token + isi secret

1. Buka <https://www.kaggle.com/settings/api> (login `izdiharwibowo`).
2. Di bagian **API**, klik **Generate New Token** (atau **Create Legacy API Key**
   kalau yang muncul file `kaggle.json`). Salin nilainya — bentuknya seperti
   `KGAT_...`. **Jangan tempel ke chat atau ke file di repo.**
3. Buka repo GitHub → **Settings** → **Secrets and variables** → **Actions** →
   **New repository secret**.
   - Name: `KAGGLE_API_TOKEN`
   - Secret: tempel token tadi.
4. Selesai. Token cuma disimpan di GitHub Secrets (tidak terbaca publik, tidak
   ikut ter-clone).

## Cara pakai sehari-hari

```bash
# 1. edit notebook
notebooks/dasbor_kaggle.ipynb

# 2. commit + push seperti biasa
git add notebooks/dasbor_kaggle.ipynb
git commit -m "Notebook: ..."
git push
```

Workflow jalan sendiri. Cek hasilnya di tab **Actions** repo
(`Sync dashboard notebook to Kaggle`), lalu buka halaman notebook Kaggle dan
lihat nomor versi bertambah di tab **Versions**.

Mau menjalankan ulang tanpa mengubah file: tab **Actions** → pilih workflow →
**Run workflow**.

## Kalau mau dari laptop (opsional, tidak disarankan saat disk penuh)

```powershell
pip install kaggle
$env:KAGGLE_API_TOKEN = "KGAT_..."   # jangan disimpan permanen
kaggle kernels push -p notebooks --no-run
```

## Catatan

- `notebooks/kernel-metadata.json` mengunci tujuan push: `id`
  `izdiharwibowo/dasbor-tj-vs-non-tj`, publik, GPU, internet aktif.
  **Jangan ganti `id`/`title`** kecuali memang mau rename notebook (slug ikut
  berubah dan link lama mati).
- Notebook di repo sengaja tanpa output tersimpan (0 output), jadi Quick Save
  tidak menghapus hasil run apa pun.
- Kalau token bocor: hapus token di halaman settings Kaggle, buat baru, lalu
  ganti secret `KAGGLE_API_TOKEN` di GitHub.
- Workflow hanya jalan untuk perubahan di `notebooks/`; commit kode biasa tidak
  memicu push notebook.
