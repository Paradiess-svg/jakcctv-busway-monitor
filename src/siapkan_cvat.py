"""siapkan_cvat.py -- Tugas 3.7: buat paket upload CVAT (gambar + pre-label) untuk QC manual.

Hanya MENYALIN/MEN-ZIP; label & gambar asli tidak diubah.
Keluaran di <out>/ :
  label_cvat.json                 -> tempel di Project CVAT (Constructor -> Raw), urutan = indeks kelas YOLO
  <task>_gambar.zip               -> diupload saat membuat Task
  <task>_anotasi_yolo11.zip       -> diupload ke Task: Actions -> Upload annotations -> format "YOLO 1.1"
  daftar_task.txt                 -> ringkasan isi tiap task

Task (urutan pengerjaan QC):
  1_valid      200 gambar  koreksi PENUH
  2_test       200 gambar  koreksi PENUH
  3_train_bus  gambar train yang ada kotak bus (tj_bus/bus_lain)  -> prioritas train
  4_train_sisa gambar train lainnya                               -> kalau waktu cukup

Contoh:
  python src/siapkan_cvat.py --help
  python src/siapkan_cvat.py
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMA_KELAS = ["tj_bus", "bus_lain", "car", "motorcycle", "truck", "mikrotrans", "kend_prioritas"]
WARNA = ["#1f77ff", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2", "#ffff00"]


def ada_bus(lab: Path) -> bool:
    return any(l.split()[0] in ("0", "1") for l in lab.read_text(encoding="utf-8").splitlines() if l.strip())


def tulis_task(out: Path, nama: str, gambar: list, lbl_dir: Path):
    with zipfile.ZipFile(out / f"{nama}_gambar.zip", "w", zipfile.ZIP_STORED) as z:  # jpg sudah terkompresi
        for g in gambar:
            z.write(g, g.name)
    nama_ok = len(NAMA_KELAS)
    with zipfile.ZipFile(out / f"{nama}_anotasi_yolo11.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(NAMA_KELAS) + "\n")
        z.writestr("obj.data", f"classes = {nama_ok}\nnames = data/obj.names\ntrain = data/train.txt\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{g.name}\n" for g in gambar))
        for g in gambar:
            lab = lbl_dir / (g.stem + ".txt")
            z.writestr(f"obj_train_data/{g.stem}.txt", lab.read_text(encoding="utf-8") if lab.exists() else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="outputs/kaggle_upload/images", help="folder berisi <split>/*.jpg")
    ap.add_argument("--labels", default="hasil_3_6/labels_v2", help="folder berisi <split>/*.txt (hasil 3.6)")
    ap.add_argument("--out", default="outputs/cvat")
    a = ap.parse_args()

    img, lbl, out = ROOT / a.images, ROOT / a.labels, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    (out / "label_cvat.json").write_text(json.dumps(
        [{"name": n, "color": c, "type": "rectangle", "attributes": []} for n, c in zip(NAMA_KELAS, WARNA)],
        indent=2), encoding="utf-8")

    train = sorted((img / "train").glob("*.jpg"))
    train_bus = [g for g in train if ada_bus(lbl / "train" / (g.stem + ".txt"))]
    set_bus = set(train_bus)
    task = {
        "1_valid": (sorted((img / "valid").glob("*.jpg")), lbl / "valid"),
        "2_test": (sorted((img / "test").glob("*.jpg")), lbl / "test"),
        "3_train_bus": (train_bus, lbl / "train"),
        "4_train_sisa": ([g for g in train if g not in set_bus], lbl / "train"),
    }
    ringkas = []
    for nama, (gambar, ldir) in task.items():
        if not gambar:
            raise SystemExit(f"Task {nama} kosong -- cek --images / --labels")
        tulis_task(out, nama, gambar, ldir)
        ringkas.append(f"{nama}: {len(gambar)} gambar")
        print(f"  {nama}: {len(gambar)} gambar -> {nama}_gambar.zip + {nama}_anotasi_yolo11.zip")
    (out / "daftar_task.txt").write_text("\n".join(ringkas) + "\n", encoding="utf-8")
    print(f"Selesai. Paket di {out}")


if __name__ == "__main__":
    main()
