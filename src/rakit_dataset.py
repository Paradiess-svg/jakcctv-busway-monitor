"""rakit_dataset.py -- Tugas 3.10: rakit dataset YOLO final dari hasil QC CVAT.

Sumber (tidak diubah):
  outputs/kaggle_upload/images/<split>/*.jpg     gambar (640x360)
  outputs/kaggle_upload/manifest.csv             split, kamera, tanggal, jam
  outputs/dataset_v1_frames.csv                  kolom siang_malam
  hasil_qc/{1_valid,2_test,3_train,4_train_sisa}/obj_train_data/*.txt   label hasil QC (YOLO 1.1)

Keluaran <out>/ (default data/dataset/v2):
  images/{train,valid,test}/*.jpg, labels/{train,valid,test}/*.txt
  data.yaml                    siap dipakai ultralytics (train = semua gambar latih kecuali --kecualikan)
  {valid,test}_{siang,malam}.txt   daftar gambar untuk evaluasi terpisah siang vs malam
  meta.csv                     file, split, kamera, tanggal, jam, siang_malam
  dataset_stats.csv            jumlah gambar & kotak per split/kamera/siang_malam/kelas
Opsi --zip membuat <out>.zip untuk diupload ke Kaggle.

Contoh:
  python src/rakit_dataset.py --help
  python src/rakit_dataset.py --zip
"""
import argparse
import csv
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMA_KELAS = ["tj_bus", "bus_lain", "car", "motorcycle", "truck", "mikrotrans", "kend_prioritas"]
TASK = {"1_valid": "valid", "2_test": "test", "3_train": "train", "4_train_sisa": "train"}


def baca_label(f: Path) -> list:
    """Baca & validasi satu file label YOLO; error kalau ada baris rusak."""
    rows = []
    for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        p = l.split()
        if not p:
            continue
        ok = len(p) == 5 and p[0].isdigit() and int(p[0]) < len(NAMA_KELAS) \
            and all(0 <= float(t) <= 1.0001 for t in p[1:]) and float(p[3]) > 0 and float(p[4]) > 0
        if not ok:
            raise SystemExit(f"Label rusak {f}:{i}: {l!r}")
        rows.append(" ".join(p))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="outputs/kaggle_upload/images")
    ap.add_argument("--manifest", default="outputs/kaggle_upload/manifest.csv")
    ap.add_argument("--frames", default="outputs/dataset_v1_frames.csv")
    ap.add_argument("--qc", default="hasil_qc")
    ap.add_argument("--out", default="data/dataset/v2")
    ap.add_argument("--zip", action="store_true", help="buat <out>.zip untuk Kaggle")
    ap.add_argument("--kecualikan", default="config/dikecualikan_v2.txt",
                    help="daftar nama gambar (tanpa .jpg) yang tidak dimasukkan; '' = tidak ada")
    a = ap.parse_args()

    img_dir, out = ROOT / a.images, ROOT / a.out
    sm = {r["path"]: r["siang_malam"] for r in csv.DictReader(open(ROOT / a.frames, encoding="utf-8"))}
    meta = {}  # stem -> dict(split, kamera, tanggal, jam, siang_malam)
    for r in csv.DictReader(open(ROOT / a.manifest, encoding="utf-8")):
        meta[Path(r["file"]).stem] = {"split": r["split"], "kamera": r["kamera"], "tanggal": r["tanggal"],
                                      "jam": r["jam"], "siang_malam": sm[r["path_asli"]]}

    # kumpulkan label QC per task, cek tidak ada gambar ganda / hilang / salah split
    label, asal = {}, {}
    for task, split in TASK.items():
        d = ROOT / a.qc / task / "obj_train_data"
        files = sorted(d.glob("*.txt"))
        if not files:
            raise SystemExit(f"Tidak ada label di {d}")
        for f in files:
            if f.stem in label:
                raise SystemExit(f"{f.stem} muncul di dua task ({asal[f.stem]} & {task})")
            if f.stem not in meta:
                raise SystemExit(f"{f.stem} tidak ada di manifest")
            if meta[f.stem]["split"] != split:
                raise SystemExit(f"{f.stem}: split manifest {meta[f.stem]['split']} != task {task}")
            label[f.stem], asal[f.stem] = baca_label(f), task
    hilang = set(meta) - set(label)
    if hilang:
        raise SystemExit(f"{len(hilang)} gambar belum punya label QC, mis. {sorted(hilang)[:3]}")

    # gambar yang dikecualikan (mis. kotak dobel beda kelas yang tidak sempat diputuskan)
    kecuali = set()
    if a.kecualikan:
        kecuali = {l.strip() for l in open(ROOT / a.kecualikan, encoding="utf-8")
                   if l.strip() and not l.startswith("#")}
        tak_dikenal = kecuali - set(meta)
        if tak_dikenal:
            raise SystemExit(f"{len(tak_dikenal)} nama di {a.kecualikan} tidak ada di manifest, mis. {sorted(tak_dikenal)[:3]}")
        for s in kecuali:
            label.pop(s)
            meta.pop(s)
    print(f"Dikecualikan: {len(kecuali)} gambar ({a.kecualikan or '-'})")

    # salin gambar + tulis label
    if out.exists():
        shutil.rmtree(out)
    for stem, rows in label.items():
        split = meta[stem]["split"]
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_dir / split / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg")
        (out / "labels" / split / f"{stem}.txt").write_text("".join(r + "\n" for r in rows), encoding="utf-8")

    # data.yaml (tanpa 'path': ultralytics memakai folder tempat yaml ini)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMA_KELAS))
    (out / "data.yaml").write_text(
        "# Dataset TJ v2: " + ", ".join(f"{sp} {sum(m['split'] == sp for m in meta.values())}" for sp in ("train", "valid", "test"))
        + f" (QC penuh; {len(kecuali)} gambar berkotak dobel dikecualikan). Split per kamera-tanggal.\n"
        "train: images/train\nval: images/valid\ntest: images/test\n"
        f"nc: {len(NAMA_KELAS)}\nnames:\n{names}\n", encoding="utf-8")

    # daftar siang/malam (path './' = relatif ke folder file txt, dipahami ultralytics)
    for split in ("valid", "test"):
        for waktu in ("siang", "malam"):
            stems = sorted(s for s, m in meta.items() if m["split"] == split and m["siang_malam"] == waktu)
            (out / f"{split}_{waktu}.txt").write_text(
                "".join(f"./images/{split}/{s}.jpg\n" for s in stems), encoding="utf-8")

    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "split", "kamera", "tanggal", "jam", "siang_malam"])
        for s in sorted(meta):
            m = meta[s]
            w.writerow([s, m["split"], m["kamera"], m["tanggal"], m["jam"], m["siang_malam"]])

    # statistik
    stat = defaultdict(lambda: [0, Counter()])
    for s, m in meta.items():
        k = stat[(m["split"], m["kamera"], m["siang_malam"])]
        k[0] += 1
        k[1].update(int(r.split()[0]) for r in label[s])
    with open(out / "dataset_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "kamera", "siang_malam", "gambar", "kotak"] + NAMA_KELAS)
        for key in sorted(stat):
            n, c = stat[key]
            w.writerow(list(key) + [n, sum(c.values())] + [c.get(i, 0) for i in range(len(NAMA_KELAS))])

    for split in ("train", "valid", "test"):
        n, c = 0, Counter()
        for key, (m, cc) in stat.items():
            if key[0] == split:
                n += m
                c += cc
        print(f"{split:6s} {n:5d} gambar | " + ", ".join(f"{NAMA_KELAS[i]} {c.get(i, 0)}" for i in range(len(NAMA_KELAS))))

    if a.zip:
        zp = out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out.parent).as_posix(),
                            compress_type=zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
        print(f"Zip: {zp} ({zp.stat().st_size / 1e6:.0f} MB)")
    print(f"Selesai -> {out}")


if __name__ == "__main__":
    main()
