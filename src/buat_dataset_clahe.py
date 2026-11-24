"""buat_dataset_clahe.py -- dataset v2 yang gambarnya sudah diproses Tahap 2 (CLAHE + gamma malam), label SAMA.

Dipakai untuk melatih model (d) = YOLO + preprocessing, sesuai alur proposal: CCTV -> CLAHE & Gamma -> YOLO.
Fungsi yang dipakai = src/preprocessing.py process() dengan setting config.yaml bagian 'preprocessing',
jadi saat inference (monitor.py, monitor.preprocessing: true) prosesnya identik (aturan #4).
Siang/malam diambil dari kolom jam di meta.csv.

Contoh:
  python src/buat_dataset_clahe.py --zip
  -> data/dataset/v2_clahe/ (+ v2_clahe.zip) dan outputs/preprocessing_ivan/contoh_v2_clahe.jpg
"""
import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import process  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/dataset/v2")
    ap.add_argument("--out", default="data/dataset/v2_clahe")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    src, out = ROOT / a.src, ROOT / a.out
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
    meta = list(csv.DictReader(open(src / "meta.csv", encoding="utf-8")))
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / "labels", out / "labels")
    for f in ("meta.csv", "dataset_stats.csv", "valid_siang.txt", "valid_malam.txt", "test_siang.txt", "test_malam.txt"):
        shutil.copy2(src / f, out / f)
    yaml_txt = (src / "data.yaml").read_text(encoding="utf-8").replace("# Dataset TJ v2:", "# Dataset TJ v2_clahe:", 1)
    p = cfg["preprocessing"]
    yaml_txt = yaml_txt.replace("\n", f" Gambar diproses CLAHE {p['clahe']['clip_limit']}/{p['clahe']['tile']}"
                                f" + gamma malam {p['gamma_night']['value']} (jam >= {p['malam_mulai_jam']} atau < {p['malam_selesai_jam']}).\n", 1)
    (out / "data.yaml").write_text(yaml_txt, encoding="utf-8")
    n_malam, contoh = 0, []
    for i, r in enumerate(meta):
        d = out / "images" / r["split"]
        d.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src / "images" / r["split"] / f"{r['file']}.jpg"))
        hasil, waktu = process(img, cfg, r["jam"])
        n_malam += waktu == "night"
        cv2.imwrite(str(d / f"{r['file']}.jpg"), hasil, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if (waktu == "night") != (r["siang_malam"] == "malam"):
            raise SystemExit(f"{r['file']}: siang/malam dari jam tidak cocok dengan label dataset")
        if i % 200 == 0 and len(contoh) < 6:
            contoh.append(np.hstack([img, hasil]))
    cv2.imwrite(str(ROOT / "outputs" / "preprocessing_ivan" / "contoh_v2_clahe.jpg"), np.vstack(contoh),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(meta)} gambar diproses ({n_malam} malam pakai gamma) -> {out}")
    if a.zip:
        zp = out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out.parent).as_posix(),
                            compress_type=zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
        print(f"Zip: {zp} ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
