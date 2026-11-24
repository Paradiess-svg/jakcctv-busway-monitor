"""autolabel.py -- Tugas 3.2/3.5: pre-label otomatis (format YOLO) dengan YOLO11x pretrained COCO.

Dirancang untuk dijalankan di Kaggle (GPU), tapi jalan juga di laptop (lambat, CPU).
Masukan : folder gambar, boleh berisi subfolder split (train/valid/test).
Keluaran: <out>/labels/<split>/<nama>.txt   satu file per gambar (kosong = tidak ada objek)
          <out>/autolabel_conf.csv          confidence tiap kotak, untuk QC (cek yang rendah dulu)
          <out>/classes.txt                 daftar kelas sesuai indeks
          <out>/ringkasan.txt               jumlah kotak per kelas per split

Kelas kita (indeks YOLO):  0 tj_bus | 1 bus_lain | 2 car | 3 motorcycle | 4 truck | 5 mikrotrans | 6 kend_prioritas
Pemetaan COCO -> kita    :  car(2)->2, motorcycle(3)->3, bus(5)->1, truck(7)->4
SEMUA bus sementara = 1 (bus_lain). Pemilahan TJ vs bukan TJ dikerjakan di 3.6 (CLIP + K-Means).
Kelas 5/6 (keputusan Dihar 10 Nov) TIDAK pernah keluar dari autolabel (COCO tidak kenal); diisi saat 3.6/QC.

Ini PRE-LABEL, bukan kebenaran: kotak valid/test wajib dikoreksi manusia (3.7). Jangan klaim ini ground truth.

Contoh (Kaggle):
  !pip -q install ultralytics
  !python autolabel.py --images /kaggle/input/NAMA-DATASET/images --out /kaggle/working/autolabel_out --limit 20   # uji dulu
  !python autolabel.py --images /kaggle/input/NAMA-DATASET/images --out /kaggle/working/autolabel_out             # semua
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

COCO_KE_KITA = {2: 2, 3: 3, 5: 1, 7: 4}
NAMA_KELAS = ["tj_bus", "bus_lain", "car", "motorcycle", "truck", "mikrotrans", "kend_prioritas"]
EKSTENSI = {".jpg", ".jpeg", ".png"}


def kumpulkan_gambar(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EKSTENSI)


def split_dari_path(p: Path, root: Path) -> str:
    rel = p.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "all"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder gambar (boleh berisi subfolder train/valid/test)")
    ap.add_argument("--out", required=True, help="folder keluaran")
    ap.add_argument("--model", default="yolo11x.pt", help="bobot YOLO (default yolo11x.pt, diunduh otomatis)")
    ap.add_argument("--conf", type=float, default=0.2, help="ambang confidence (turunkan ke 0.15 kalau motor banyak terlewat)")
    ap.add_argument("--iou", type=float, default=0.6, help="ambang IoU untuk NMS")
    ap.add_argument("--imgsz", type=int, default=960, help="ukuran input; lebih besar = objek kecil/jauh lebih terdeteksi tapi lebih lambat")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="mis. 0 (GPU) atau cpu; default otomatis")
    ap.add_argument("--agnostic-nms", action="store_true", help="NMS lintas kelas (kurangi kotak car+truck ganda)")
    ap.add_argument("--limit", type=int, default=0, help="proses hanya N gambar pertama (untuk uji)")
    args = ap.parse_args()

    from ultralytics import YOLO  # import di sini supaya --help jalan tanpa ultralytics

    root, out = Path(args.images), Path(args.out)
    gambar = kumpulkan_gambar(root)
    if args.limit:
        gambar = gambar[: args.limit]
    if not gambar:
        raise SystemExit(f"Tidak ada gambar di {root}")
    print(f"{len(gambar)} gambar, model {args.model}, conf {args.conf}, imgsz {args.imgsz}")

    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(NAMA_KELAS) + "\n", encoding="utf-8")
    model = YOLO(args.model)

    conf_rows = []
    hitung = defaultdict(Counter)   # split -> kelas -> jumlah
    kosong = Counter()              # split -> jumlah gambar tanpa kotak
    for i in range(0, len(gambar), args.batch):
        potong = gambar[i : i + args.batch]
        hasil = model.predict(
            source=[str(p) for p in potong], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            classes=list(COCO_KE_KITA), device=args.device, agnostic_nms=args.agnostic_nms,
            verbose=False,
        )
        for p, r in zip(potong, hasil):
            sp = split_dari_path(p, root)
            baris = []
            for k, (xywhn, c, cf) in enumerate(zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                kelas = COCO_KE_KITA[int(c)]
                baris.append(f"{kelas} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                conf_rows.append([sp, p.name, k, kelas, NAMA_KELAS[kelas], f"{cf:.4f}"])
                hitung[sp][NAMA_KELAS[kelas]] += 1
            if not baris:
                kosong[sp] += 1
            lab = out / "labels" / sp / (p.stem + ".txt")
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("\n".join(baris) + ("\n" if baris else ""), encoding="utf-8")
        if (i // args.batch) % 10 == 0:
            print(f"  {min(i + args.batch, len(gambar))}/{len(gambar)}")

    with open(out / "autolabel_conf.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "gambar", "baris", "kelas_id", "kelas", "confidence"])
        w.writerows(conf_rows)

    teks = []
    for sp in sorted(hitung.keys() | kosong.keys()):
        teks.append(f"[{sp}] kotak: {dict(hitung[sp])} | gambar tanpa kotak: {kosong[sp]}")
    (out / "ringkasan.txt").write_text("\n".join(teks) + "\n", encoding="utf-8")
    print("\n".join(teks))
    print(f"Selesai. Label di {out/'labels'}, confidence di {out/'autolabel_conf.csv'}")


if __name__ == "__main__":
    main()
