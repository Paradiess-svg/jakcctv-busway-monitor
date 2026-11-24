"""bersihkan_dobel.py -- buang kotak label DOBEL di hasil_qc/ (posisi hampir sama, IoU > --iou).

Penyebab: pre-label task 4 ter-impor dua kali di CVAT (IoU 1,0), dan sebagian edit CVAT meninggalkan salinan.
Aturan per kelompok kotak dobel:
  - semua kelas sama            -> simpan satu
  - kelas berbeda               -> (tanpa --hanya-sama) simpan kelas yang BERBEDA dari pre-label. PERINGATAN 16 Nov:
                                   aturan ini TIDAK andal (kotak dobel sudah ada sejak autolabel tanpa agnostic NMS);
                                   pakai --hanya-sama dan putuskan yang beda kelas secara manual.
                                   pre-label diambil dari outputs/cvat/<task>_anotasi_yolo11.zip
  - tidak bisa diputuskan       -> semua kotak dibiarkan, dicatat di laporan untuk dicek manual
Label lama dicadangkan ke outputs/cadangan_dobel_<waktu>/ sebelum ditulis.

Contoh:
  python src/bersihkan_dobel.py            # dry run: hanya laporan
  python src/bersihkan_dobel.py --tulis
"""
import argparse
import csv
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluasi import NAMA, iou  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASK = {"1_valid": "1_valid", "2_test": "2_test", "3_train": "3_train_bus", "4_train_sisa": "4_train_sisa"}


def kotak(baris):
    c, x, y, w, h = int(baris[0]), *map(float, baris[1:5])
    return c, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--tulis", action="store_true")
    ap.add_argument("--hanya-sama", action="store_true",
                    help="hanya buang dobel berkelas SAMA; dobel beda kelas cuma dilaporkan (aturan pre-label terbukti tidak andal)")
    a = ap.parse_args()

    laporan, total, ditulis = [], Counter(), {}
    for task, zipnama in TASK.items():
        with zipfile.ZipFile(ROOT / "outputs" / "cvat" / f"{zipnama}_anotasi_yolo11.zip") as z:
            pre = {Path(n).stem: [kotak(l.split()) for l in z.read(n).decode().splitlines() if l.strip()]
                   for n in z.namelist() if n.startswith("obj_train_data/")}
        for f in sorted((ROOT / "hasil_qc" / task / "obj_train_data").glob("*.txt")):
            rows = [l.split() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            bx = [kotak(r) for r in rows]
            # kelompokkan kotak yang saling dobel (union-find sederhana)
            induk = list(range(len(bx)))

            def akar(i):
                while induk[i] != i:
                    i = induk[i]
                return i
            for i in range(len(bx)):
                for j in range(i + 1, len(bx)):
                    if iou(bx[i][1], bx[j][1]) > a.iou:
                        induk[akar(j)] = akar(i)
            grup = {}
            for i in range(len(bx)):
                grup.setdefault(akar(i), []).append(i)
            buang = set()
            for anggota in grup.values():
                if len(anggota) < 2:
                    continue
                kelas = {bx[i][0] for i in anggota}
                if len(kelas) == 1:
                    buang |= set(anggota[1:])
                    total[f"{task}: sama"] += len(anggota) - 1
                    continue
                # kelas pre-label di posisi ini
                cand = [(iou(bx[anggota[0]][1], p[1]), p[0]) for p in pre.get(f.stem, [])]
                pc = max(cand)[1] if cand and max(cand)[0] > a.iou else None
                beda = {c for c in kelas if c != pc} if pc is not None else set()
                nama = "/".join(sorted(NAMA[c] for c in kelas))
                if a.hanya_sama:
                    total[f"{task}: beda (dilaporkan saja)"] += 1
                    laporan.append([task, f.stem, nama, NAMA[pc] if pc is not None else "-", "", "CEK MANUAL"])
                elif pc is not None and len(beda) == 1:
                    pilih = beda.pop()
                    simpan = next(i for i in anggota if bx[i][0] == pilih)
                    buang |= {i for i in anggota if i != simpan}
                    total[f"{task}: beda -> {NAMA[pilih]}"] += 1
                    laporan.append([task, f.stem, nama, NAMA[pc], NAMA[pilih], "otomatis"])
                else:
                    total[f"{task}: beda TIDAK diputuskan"] += 1
                    laporan.append([task, f.stem, nama, NAMA[pc] if pc is not None else "-", "", "CEK MANUAL"])
            if buang:
                ditulis[f] = [r for i, r in enumerate(rows) if i not in buang]

    for k in sorted(total):
        print(f"  {k}: {total[k]}")
    out = ROOT / "outputs" / "laporan_dobel.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "file", "kelas_dobel", "kelas_prelabel", "kelas_disimpan", "status"])
        w.writerows(laporan)
    print(f"{len(ditulis)} file berubah. Rincian kelas-beda: {out}")
    if not a.tulis:
        print("Dry run. Tambah --tulis untuk menulis.")
        return
    cad = ROOT / "outputs" / f"cadangan_dobel_{datetime.now():%Y%m%d_%H%M%S}"
    for f, rows in ditulis.items():
        (cad / f.parent.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, cad / f.parent.parent.name / f.name)
        f.write_text("".join(" ".join(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Ditulis. Cadangan: {cad}")


if __name__ == "__main__":
    main()
