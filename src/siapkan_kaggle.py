"""siapkan_kaggle.py -- salin frame terpilih (dataset_v1_frames.csv) ke satu folder untuk di-upload ke Kaggle.

Hanya MENYALIN; data/raw tidak diubah/dihapus. Nama file dibuat unik:
  outputs/kaggle_upload/images/<split>/<kamera>__<tanggal>__<namafile>.jpg
Juga menulis outputs/kaggle_upload/manifest.csv (nama baru <-> path asli, split, kamera, tanggal, jam)
dan (opsional) satu zip.

Contoh:
  python src/siapkan_kaggle.py --help
  python src/siapkan_kaggle.py            # salin saja
  python src/siapkan_kaggle.py --zip      # salin + buat outputs/kaggle_upload.zip
"""
import argparse
import csv
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/dataset_v1_frames.csv")
    ap.add_argument("--out", default="outputs/kaggle_upload")
    ap.add_argument("--zip", action="store_true", help="buat juga <out>.zip")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / args.csv, encoding="utf-8")))
    out = ROOT / args.out
    # cek ruang disk (butuh ±ukuran total x 2 kalau --zip)
    total = sum((ROOT / r["path"]).stat().st_size for r in rows)
    free = shutil.disk_usage(ROOT).free
    butuh = total * (2.2 if args.zip else 1.1)
    print(f"{len(rows)} frame, {total/1e6:.0f} MB; disk kosong {free/1e9:.1f} GB")
    if free < butuh + 1e9:
        raise SystemExit("Ruang disk tidak cukup (sisakan >= 1 GB). Dibatalkan, tidak ada yang disalin.")

    manifest, baru, lewat = [], 0, 0
    for r in rows:
        src = ROOT / r["path"]
        nama = f'{r["kamera"]}__{r["tanggal"]}__{src.name}'
        dst = out / "images" / r["split"] / nama
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            lewat += 1
        else:
            shutil.copy2(src, dst)
            baru += 1
        manifest.append({"file": f'{r["split"]}/{nama}', "path_asli": r["path"], "split": r["split"],
                         "kamera": r["kamera"], "tanggal": r["tanggal"], "jam": r["jam"],
                         "periode": r["periode"], "idle": r["idle"]})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"Disalin {baru}, sudah ada {lewat}. Folder: {out}")

    if args.zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # jpg sudah terkompres
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        print(f"Zip: {zpath} ({zpath.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
