# API dasbor monitoring busway (untuk tim desain/frontend)

Server: `python dashboard/server.py` (dari folder `Project-TJ`) → `http://localhost:8000`.
Dokumentasi interaktif (bisa dicoba langsung): `http://localhost:8000/docs`.
Halaman uji sederhana (bukan desain final): `http://localhost:8000/`. Contoh pemakaian semua endpoint ada di `dashboard/static/index.html`.

Sistem ini hanya **mendeteksi dan mencatat**, tidak menilang. Ada dua kelompok log:
- `tj`: `tj_bus`, `mikrotrans`
- `non_tj`: `bus_lain`, `car`, `motorcycle`, `truck`

`kend_prioritas` tidak dicatat. Kendaraan dicatat sekali, setelah ≥ 2 detik berada di ROI (jalur busway).

## Umum
| Method | URL | Isi |
|---|---|---|
| GET | `/api/kamera` | `[{id, nama, roi}]`: kamera yang tersedia (`roi: true` = bisa dipakai) |
| GET | `/api/kamera/{id}/roi` | poligon ROI, koordinat frame 640×360 (`polygons: [[[x,y],...]]`) |
| GET | `/api/config` | aturan yang dipakai (model, conf, 2 detik, pembagian kelompok) |

## Mode live (grid maks 4 kamera)
| Method | URL | Isi |
|---|---|---|
| POST | `/api/live/start` | body JSON `{"kamera": ["c01_taman_literasi", ...]}` (1–4). Butuh ±5 detik (memuat model) |
| POST | `/api/live/stop` | hentikan |
| GET | `/api/live/status` | per kamera: `stream` (`menyambung`/`hidup`/`putus`), `fps_analisis`, `hitung {tj, non_tj}` |
| GET | `/api/live/{id}/mjpeg` | video beranotasi, tinggal pasang `<img src="/api/live/c01_taman_literasi/mjpeg">` |
| GET | `/api/live/{id}/frame.jpg` | satu frame beranotasi terbaru (alternatif kalau MJPEG tidak cocok) |
| GET | `/api/live/log?setelah_id=0&kelompok=tj` | kejadian baru. Polling tiap 1–2 detik, kirim `setelah_id` = `id` terbesar yang sudah diterima |

Satu kejadian (log):
```json
{"id": 6, "waktu": "2026-11-17 17:23:06", "kamera": "c01_taman_literasi", "track_id": 93,
 "kelas": "tj_bus", "kelompok": "tj", "conf_rata2": 0.717, "detik_di_roi": 2.0,
 "snapshot_url": "/files/live/c01_taman_literasi/2026-11-17_172224/snapshot/20261117_172306_tj_tj_bus_id93.jpg"}
```
Warna kotak di video: hijau = TJ, merah = non-TJ, abu-abu = di luar ROI, garis kuning = ROI.

## Mode upload
| Method | URL | Isi |
|---|---|---|
| POST | `/api/upload` | form-data: `file` (mp4/avi/mov/mkv/ts, maks 500 MB) + `kamera` (ID kamera; menentukan ROI). Balik `{job_id}` |
| GET | `/api/upload/{job_id}` | `status` (`proses`/`selesai`/`gagal`), `progres` 0–1, `hitung`, `kejadian[]`, setelah selesai: `video_hasil` (mp4 H.264, bisa diputar `<video>`), `ringkasan` |
| GET | `/api/upload/{job_id}/preview.jpg` | frame beranotasi terakhir selama proses |
| GET | `/api/upload` | daftar job |

Waktu proses di laptop (CPU): ±0,3–0,7 detik per frame, dianalisis 5 frame per detik video. Jadi video 1 menit butuh ±2–3 menit.

## Riwayat, verifikasi, akurasi (19 Nov)
| Method | URL | Isi |
|---|---|---|
| GET | `/api/log?kelompok=tj&kamera=&tanggal=YYYY-MM-DD&verifikasi=belum&batas=50&offset=0` | riwayat semua sesi live (dari CSV), terbaru dulu |
| POST | `/api/verifikasi` | `{"id": "<id log>", "status": "benar"/"salah"/null}` (tanda cek manusia) |
| GET | `/api/akurasi` | % benar dari verifikasi manusia (semua, per kelompok, per kamera) + angka uji model di data test |
| GET | `/api/ringkasan?periode=hari_ini` | angka kartu dasbor (total, non-TJ, sterilisasi %, kamera online) + per jam/kelas/kamera |
`/api/live/status` per kamera juga berisi `lampu` (merah/hijau/kuning/mati) dan `kepadatan` (lancar/sedang/padat).

## Statistik
`GET /api/statistik?sumber=live` atau `?sumber=<job_id>` →
`{total: {tj, non_tj}, per_kelas: {...}, per_kamera: {...}, per_jam: [{jam, kelompok, jumlah}]}`

## Batasan yang perlu ditampilkan jujur di UI
- Live 4 kamera di laptop CPU: ±0,75 frame dianalisis per detik per kamera. Kalau 1–2 kamera, jadi lebih cepat.
- Video live tertinggal beberapa detik dari kondisi nyata, karena stream HLS JakCCTV memang tertunda.
