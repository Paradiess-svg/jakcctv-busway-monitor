"""evaluasi.py -- Tugas 4.x / 7.3: evaluasi model pada confidence OPERASIONAL (bukan conf 0,001 bawaan val).

Melengkapi metrik_per_kelas.csv (mAP dari Kaggle) dengan hal yang dipakai sistem pelanggaran:
  1. Confusion matrix 7 kelas + background pada --conf (default 0,25), per split dan siang/malam.
  2. Metrik biner TJ vs bukan-TJ:
       - TJ terbaca bukan-TJ  -> calon PELANGGARAN PALSU
       - bukan-TJ terbaca TJ  -> PELANGGARAN LOLOS
  3. Gambar contoh: prediksi vs label (acak siang/malam) dan semua kasus salah TJ.
  4. Waktu inference rata-rata per gambar di mesin ini (gambaran awal benchmark CPU 4.6).

Pencocokan: tiap gambar, pasangan (label, prediksi) diurutkan IoU menurun, dipasangkan 1-1 bila IoU >= --iou,
tanpa melihat kelas (supaya salah-kelas terlihat di matriks). Prediksi tanpa pasangan = background (FP),
label tanpa pasangan = terlewat (FN).

Keluaran <out>/:
  confusion_<subset>.csv / .png       subset = valid, test, test_siang, test_malam, ...
  tj_vs_bukan.csv                     ringkasan biner per subset
  per_kelas_conf.csv                  precision/recall per kelas pada --conf
  contoh/*.jpg, salah_tj/*.jpg        kiri = label, kanan = prediksi
  tj_tanpa_label/*.jpg                model memprediksi TJ di tempat yang tidak ada labelnya
  ringkasan.txt

Contoh:
  python src/evaluasi.py --help
  python src/evaluasi.py --model "hasil_training_v2/runs/tj_v2/weights/best.pt"
"""
import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMA = ["tj_bus", "bus_lain", "car", "motorcycle", "truck", "mikrotrans", "kend_prioritas"]
BG = len(NAMA)  # indeks background
WARNA = [(255, 120, 30), (30, 130, 255), (60, 180, 60), (40, 40, 220), (180, 80, 150), (200, 120, 230), (0, 230, 230)]


def baca_gt(f: Path, w: int, h: int) -> list:
    """Label YOLO -> [(kelas, x1, y1, x2, y2)] dalam piksel."""
    out = []
    if f.exists():
        for l in f.read_text(encoding="utf-8").splitlines():
            p = l.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cocokkan(gt: list, pred: list, thr: float) -> list:
    """Return daftar (kelas_gt, kelas_pred, idx_gt, idx_pred); BG dipakai untuk yang tak berpasangan."""
    pas = sorted(((iou(g[1:], p[1:5]), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    gi, pj, hasil = set(), set(), []
    for v, i, j in pas:
        if v < thr:
            break
        if i in gi or j in pj:
            continue
        gi.add(i)
        pj.add(j)
        hasil.append((gt[i][0], pred[j][0], i, j))
    hasil += [(g[0], BG, i, None) for i, g in enumerate(gt) if i not in gi]
    hasil += [(BG, p[0], None, j) for j, p in enumerate(pred) if j not in pj]
    return hasil


def gambar_kotak(img, kotak, judul):
    im = img.copy()
    for k in kotak:
        c, x1, y1, x2, y2 = k[:5]
        teks = NAMA[c] + (f" {k[5]:.2f}" if len(k) > 5 else "")
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), WARNA[c], 1)
        cv2.putText(im, teks, (int(x1), max(9, int(y1) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, WARNA[c], 1)
    cv2.putText(im, judul, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(im, judul, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return im


def simpan_cm(cm: np.ndarray, path: Path, judul: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = NAMA + ["background"]
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pred \\ label"] + lab)
        for i, r in enumerate(cm):
            w.writerow([lab[i]] + list(map(int, r)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(np.log1p(cm), cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 3 else "black")
    ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right")
    ax.set_yticks(range(len(lab)), lab)
    ax.set_xlabel("Label (ground truth)")
    ax.set_ylabel("Prediksi")
    ax.set_title(judul, fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="hasil_training_v2/runs/tj_v2/weights/best.pt")
    ap.add_argument("--data", default="data/dataset/v2", help="folder dataset hasil rakit_dataset.py")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--conf", type=float, default=0.25, help="confidence operasional")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU minimum untuk dianggap pasangan")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--contoh", type=int, default=6, help="jumlah gambar contoh per split (separuh siang, separuh malam)")
    ap.add_argument("--out", default="outputs/evaluasi_v2")
    a = ap.parse_args()

    from ultralytics import YOLO

    data, out = ROOT / a.data, ROOT / a.out
    (out / "contoh").mkdir(parents=True, exist_ok=True)
    (out / "salah_tj").mkdir(parents=True, exist_ok=True)
    (out / "tj_tanpa_label").mkdir(parents=True, exist_ok=True)
    meta = {r["file"]: r for r in csv.DictReader(open(data / "meta.csv", encoding="utf-8"))}
    model = YOLO(str(ROOT / a.model))
    model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=a.imgsz, verbose=False)  # pemanasan

    cms = defaultdict(lambda: np.zeros((BG + 1, BG + 1), int))  # subset -> matriks [pred, gt]
    waktu, n_salah = [], 0
    random.seed(0)
    for split in a.splits:
        imgs = sorted((data / "images" / split).glob("*.jpg"))
        contoh = set()
        for sm in ("siang", "malam"):
            kand = [p for p in imgs if meta[p.stem]["siang_malam"] == sm]
            contoh |= set(random.sample(kand, min(a.contoh // 2, len(kand))))
        print(f"[{split}] {len(imgs)} gambar ...", flush=True)
        for p in imgs:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
            waktu.append(time.perf_counter() - t0)
            pred = [(int(c), *map(float, b), float(s)) for b, c, s in
                    zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            gt = baca_gt(data / "labels" / split / f"{p.stem}.txt", w, h)
            pas = cocokkan(gt, pred, a.iou)
            sm = meta[p.stem]["siang_malam"]
            for g, q, _, _ in pas:
                cms[split][q, g] += 1
                cms[f"{split}_{sm}"][q, g] += 1
            # kasus salah TJ: label TJ diprediksi kelas lain/terlewat, atau kelas lain diprediksi TJ
            salah = [(g, q) for g, q, _, _ in pas if (g == 0) != (q == 0) and g != BG and q != BG]
            hantu = [1 for g, q, _, _ in pas if g == BG and q == 0]  # prediksi TJ tanpa label: TJ lupa dilabel? atau salah deteksi?
            if p in contoh or salah or hantu:
                gab = np.hstack([gambar_kotak(img, gt, "LABEL"),
                                 gambar_kotak(img, pred, f"PREDIKSI conf>={a.conf}")])
                if p in contoh:
                    cv2.imwrite(str(out / "contoh" / f"{split}_{sm}_{p.stem}.jpg"), gab)
                if salah:
                    n_salah += 1
                    ket = "_".join(f"{NAMA[g]}-jadi-{NAMA[q]}" for g, q in salah[:2])
                    cv2.imwrite(str(out / "salah_tj" / f"{split}_{sm}_{p.stem}__{ket}.jpg"), gab)
                if hantu:
                    cv2.imwrite(str(out / "tj_tanpa_label" / f"{split}_{sm}_{p.stem}__{len(hantu)}x.jpg"), gab)

    # simpan matriks + metrik
    urut = [s for sp in a.splits for s in (sp, f"{sp}_siang", f"{sp}_malam")]
    baris_tj, baris_kls = [], []
    for s in urut:
        cm = cms[s]
        simpan_cm(cm, out / f"confusion_{s}", f"{s} | conf>={a.conf}, IoU>={a.iou} | baris=prediksi, kolom=label")
        for c in range(BG):
            tp, n_pred, n_gt = cm[c, c], cm[c, :].sum(), cm[:, c].sum()
            baris_kls.append(dict(subset=s, kelas=NAMA[c], label=int(n_gt), prediksi=int(n_pred), benar=int(tp),
                                  precision=round(tp / n_pred, 3) if n_pred else "",
                                  recall=round(tp / n_gt, 3) if n_gt else ""))
        bukan = list(range(1, BG))
        tj_gt = cm[:, 0].sum()
        tj_benar = cm[0, 0]
        tj_jadi_lain = cm[bukan, 0].sum()
        tj_terlewat = cm[BG, 0]
        lain_gt = cm[:, bukan].sum()
        lain_jadi_tj = cm[0, bukan].sum()
        tj_hantu = cm[0, BG]
        lain_terdeteksi = cm[:BG, bukan].sum()
        baris_tj.append(dict(
            subset=s, tj_label=int(tj_gt), tj_benar=int(tj_benar),
            tj_jadi_bukan_tj=int(tj_jadi_lain), tj_terlewat=int(tj_terlewat),
            bukan_tj_label=int(lain_gt), bukan_tj_jadi_tj=int(lain_jadi_tj),
            prediksi_tj_tanpa_label=int(tj_hantu),
            recall_tj=round(tj_benar / tj_gt, 3) if tj_gt else "",
            precision_tj=round(tj_benar / cm[0, :].sum(), 3) if cm[0, :].sum() else "",
            persen_tj_jadi_bukan_tj=round(tj_jadi_lain / tj_gt, 3) if tj_gt else "",
            persen_bukan_tj_jadi_tj=round(lain_jadi_tj / lain_gt, 4) if lain_gt else "",
            recall_kendaraan_bukan_tj=round(lain_terdeteksi / lain_gt, 3) if lain_gt else "",
        ))
    for nama, rows in [("tj_vs_bukan.csv", baris_tj), ("per_kelas_conf.csv", baris_kls)]:
        with open(out / nama, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    ms = 1000 * np.array(waktu)
    teks = [f"Model: {a.model}", f"conf={a.conf}  IoU cocok={a.iou}  imgsz={a.imgsz}",
            f"Waktu inference CPU mesin ini: rata-rata {ms.mean():.0f} ms/gambar (median {np.median(ms):.0f}, n={len(ms)})",
            f"Gambar dengan kasus salah TJ: {n_salah} (folder salah_tj/)", "", "TJ vs bukan-TJ:"]
    for b in baris_tj:
        teks.append(f"  {b['subset']:12s} TJ {b['tj_benar']}/{b['tj_label']} benar (recall {b['recall_tj']}, precision {b['precision_tj']}); "
                    f"TJ->bukan-TJ {b['tj_jadi_bukan_tj']}, TJ terlewat {b['tj_terlewat']}; "
                    f"bukan-TJ->TJ {b['bukan_tj_jadi_tj']}/{b['bukan_tj_label']}; prediksi TJ tanpa label {b['prediksi_tj_tanpa_label']}")
    (out / "ringkasan.txt").write_text("\n".join(teks) + "\n", encoding="utf-8")
    print("\n".join(teks))
    print(f"Selesai -> {out}")


if __name__ == "__main__":
    main()
