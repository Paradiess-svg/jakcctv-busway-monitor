"""siapkan_dasbor_cloud.py -- bungkus dasbor (kode + model v2 + ROI + config) jadi satu zip untuk Kaggle/Colab.

Contoh:
  python src/siapkan_dasbor_cloud.py
  -> outputs/dasbor_cloud.zip  (upload ke Kaggle sebagai Dataset "jakcctv-dasbor", lalu jalankan
     notebooks/dasbor_kaggle.ipynb)
Log/snapshot lama TIDAK ikut (di cloud mulai dari nol).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISI = [
    "dashboard/server.py", "dashboard/live.py", "dashboard/API.md",
    "src/monitor.py", "src/capture.py", "src/preprocessing.py",
    "config/config.yaml",
    "hasil_training_v2/runs/tj_v2/weights/best.pt",
    "outputs/evaluasi_v2/tj_vs_bukan.csv",
]


def main():
    files = [ROOT / f for f in ISI]
    files += sorted((ROOT / "dashboard" / "static").rglob("*.*"))
    files += sorted((ROOT / "config" / "roi").glob("*.json"))      # _lama/ tidak ikut
    out = ROOT / "outputs" / "dasbor_cloud.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if not f.exists():
                raise SystemExit(f"tidak ada: {f}")
            z.write(f, "jakcctv_app/" + f.relative_to(ROOT).as_posix())
    print(f"{len(files)} file -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
