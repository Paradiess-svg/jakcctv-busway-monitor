"""lembar_cek_label.py -- buat gambar cek label bernomor + file isian, lalu terapkan koreksinya.

Langkah 1 (buat lembar):
  python src/lembar_cek_label.py buat
  -> outputs/evaluasi_v1/cek/<no>_<nama>.jpg : ATAS = label kita (kotak L1, L2, ...),
                                              BAWAH = tebakan model (kotak P1, P2, ...), hanya contekan
  -> outputs/evaluasi_v1/cek_label.txt        : satu baris per gambar, diisi Dihar

Format isian (boleh beberapa, pisah koma):
  ok            label sudah benar
  L3=tj_bus     ganti kelas kotak label L3
  L3=hapus      hapus kotak label L3 (bukan kendaraan / dobel)
  P2=tj_bus     tambah kotak baru, posisinya diambil dari kotak model P2, kelasnya tj_bus
  ragu          dilihat bareng nanti

Langkah 2 (terapkan, label lama dicadangkan dulu):
  python src/lembar_cek_label.py terapkan            # cek dulu tanpa mengubah (dry run)
  python src/lembar_cek_label.py terapkan --tulis    # benar-benar menulis label
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluasi import NAMA, WARNA, baca_gt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SKALA = 2  # perbesar gambar 640x360 -> 1280x720 supaya nomor terbaca


def gambar(img, kotak, awalan, judul):
    im = cv2.resize(img, None, fx=SKALA, fy=SKALA, interpolation=cv2.INTER_LINEAR)
    for n, k in enumerate(kotak, 1):
        c, x1, y1, x2, y2 = k[:5]
        x1, y1, x2, y2 = (int(v * SKALA) for v in (x1, y1, x2, y2))
        cv2.rectangle(im, (x1, y1), (x2, y2), WARNA[c], 2)
        teks = f"{awalan}{n} {NAMA[c]}"
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 3 if y1 - 3 > th + 2 else y1 + th + 3  # kotak di tepi atas: nomor ditaruh di dalam kotak
        cv2.rectangle(im, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), WARNA[c], -1)
        cv2.putText(im, teks, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    im = cv2.copyMakeBorder(im, 54, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # header di luar gambar
    cv2.putText(im, judul, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def buat(a):
    from ultralytics import YOLO
    ev, data = ROOT / a.eval, ROOT / a.data
    stems = []  # (split, stem) unik, urut
    for folder in ("salah_tj", "tj_tanpa_label"):
        for f in sorted((ev / folder).glob("*.jpg")):
            # nama = <split>_<siang|malam>_<stem>__<keterangan>.jpg ; stem sendiri mengandung '__'
            m = re.match(r"(valid|test)_(siang|malam)_(.+)__[^_]", f.stem)
            if m:
                key = (m.group(1), f.stem[m.end(2) + 1:].rsplit("__", 1)[0])
                if key not in stems:
                    stems.append(key)
    out = ev / "cek"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = YOLO(str(ROOT / a.model))
    pred_simpan, baris = {}, []
    for no, (split, stem) in enumerate(stems, 1):
        img = cv2.imread(str(data / "images" / split / f"{stem}.jpg"))
        h, w = img.shape[:2]
        gt = baca_gt(data / "labels" / split / f"{stem}.txt", w, h)
        r = model.predict(img, imgsz=640, conf=a.conf, verbose=False)[0]
        pred = [(int(c), *b) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
        pred_simpan[f"{split}/{stem}"] = pred
        lembar = np.vstack([gambar(img, gt, "L", f"ATAS = LABEL KITA (yang dinilai)   #{no:02d} {split}/{stem}"),
                            gambar(img, pred, "P", "BAWAH = TEBAKAN MODEL (contekan saja, jangan dinilai)")])
        cv2.imwrite(str(out / f"{no:02d}_{split}__{stem}.jpg"), lembar, [cv2.IMWRITE_JPEG_QUALITY, 90])
        baris.append(f"{no:02d} {split}/{stem} | ")
    (ev / "cek_pred.json").write_text(json.dumps(pred_simpan), encoding="utf-8")
    kepala = (
        "# CEK LABEL. Buka cek/<nomor>_*.jpg. ATAS = label kita (L1, L2, ...), BAWAH = tebakan model (P1, P2, ...).\n"
        "# Yang dinilai HANYA kotak di ATAS. Kesalahan model di BAWAH tidak usah diurus.\n"
        "# Isi setelah | (boleh beberapa, pisah koma):\n"
        "#   ok            label atas sudah benar\n"
        "#   L3=tj_bus     kotak L3 kelasnya salah, harusnya tj_bus\n"
        "#   L3=hapus      kotak L3 harus dibuang\n"
        "#   P2=tj_bus     ada kendaraan yang belum dikotaki di atas; pakai posisi kotak P2 dari bawah, kelas tj_bus\n"
        "#   ragu          dilihat bareng nanti\n"
        "# Kelas: " + ", ".join(NAMA) + "\n")
    (ev / "cek_label.txt").write_text(kepala + "\n".join(baris) + "\n", encoding="utf-8")
    print(f"{len(stems)} lembar -> {out}\nIsian -> {ev / 'cek_label.txt'}")


def terapkan(a):
    ev, data = ROOT / a.eval, ROOT / a.data
    pred_all = json.loads((ev / "cek_pred.json").read_text(encoding="utf-8"))
    rencana, masalah, n_ok, n_ragu, n_kosong = {}, [], 0, 0, 0
    for l in (ev / "cek_label.txt").read_text(encoding="utf-8").splitlines():
        if l.startswith("#") or "|" not in l:
            continue
        kiri, isi = l.split("|", 1)
        key = kiri.split()[1]
        isi = isi.strip().lower()
        if not isi:
            n_kosong += 1
            continue
        if isi == "ok":
            n_ok += 1
            continue
        if isi.startswith("ragu"):
            n_ragu += 1
            continue
        split, stem = key.split("/")
        f = data / "labels" / split / f"{stem}.txt"
        rows = [x.split() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        ubah, hapus, tambah = {}, set(), []
        for perintah in re.split(r"[,;]\s*", isi):
            m = re.fullmatch(r"([lp])(\d+)\s*=\s*(\w+)", perintah.strip())
            if not m:
                masalah.append(f"{key}: tidak dipahami '{perintah}'")
                continue
            jenis, n, kelas = m.group(1), int(m.group(2)), m.group(3)
            if kelas != "hapus" and kelas not in NAMA:
                masalah.append(f"{key}: kelas '{kelas}' tidak dikenal")
                continue
            if jenis == "l":
                if not 1 <= n <= len(rows):
                    masalah.append(f"{key}: L{n} tidak ada (cuma {len(rows)} kotak)")
                elif kelas == "hapus":
                    hapus.add(n - 1)
                else:
                    ubah[n - 1] = NAMA.index(kelas)
            else:
                pred = pred_all[key]
                if not 1 <= n <= len(pred) or kelas == "hapus":
                    masalah.append(f"{key}: P{n} tidak ada / tidak bisa 'hapus'")
                    continue
                img = cv2.imread(str(data / "images" / split / f"{stem}.jpg"))
                h, w = img.shape[:2]
                _, x1, y1, x2, y2 = pred[n - 1]
                tambah.append([str(NAMA.index(kelas)), f"{(x1 + x2) / 2 / w:.6f}", f"{(y1 + y2) / 2 / h:.6f}",
                               f"{(x2 - x1) / w:.6f}", f"{(y2 - y1) / h:.6f}"])
        baru = [([str(ubah[i])] + r[1:]) if i in ubah else r for i, r in enumerate(rows) if i not in hapus] + tambah
        rencana[f] = (rows, baru, isi)
    print(f"ok {n_ok}, ragu {n_ragu}, kosong {n_kosong}, diubah {len(rencana)}")
    for f, (lama, baru, isi) in rencana.items():
        print(f"  {f.parent.name}/{f.stem}: {isi}  ({len(lama)} -> {len(baru)} kotak)")
    for m in masalah:
        print("  MASALAH:", m)
    if masalah:
        print("Perbaiki isian dulu; tidak ada yang ditulis.")
        return
    if not a.tulis:
        print("Dry run. Tambah --tulis untuk menulis.")
        return
    # tulis ke dataset DAN ke sumber hasil_qc/, supaya rakit_dataset.py tidak menimpa koreksi
    qc_task = {"valid": "1_valid", "test": "2_test"}
    cad = ROOT / "outputs" / f"cadangan_label_{datetime.now():%Y%m%d_%H%M%S}"
    for f, (lama, baru, _) in rencana.items():
        split = f.parent.name
        f_qc = ROOT / "hasil_qc" / qc_task[split] / "obj_train_data" / f.name
        if sorted(" ".join(r) for r in lama) != sorted(" ".join(x.split()) for x in
                                                        f_qc.read_text(encoding="utf-8").splitlines() if x.strip()):
            raise SystemExit(f"{f_qc} beda isi dengan {f}; tidak ada yang ditulis untuk file ini dan sesudahnya")
        teks = "".join(" ".join(r) + "\n" for r in baru)
        for sumber, sub in ((f, f"dataset_{split}"), (f_qc, f"hasil_qc_{qc_task[split]}")):
            (cad / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(sumber, cad / sub / sumber.name)
            sumber.write_text(teks, encoding="utf-8")
    print(f"Ditulis {len(rencana)} gambar (data/dataset/v2/labels + hasil_qc/). Cadangan label lama: {cad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aksi", choices=["buat", "terapkan"])
    ap.add_argument("--model", default="hasil_training (NEW)/runs/tj_v1/weights/best.pt")
    ap.add_argument("--data", default="data/dataset/v2")
    ap.add_argument("--eval", default="outputs/evaluasi_v1")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tulis", action="store_true")
    a = ap.parse_args()
    buat(a) if a.aksi == "buat" else terapkan(a)


if __name__ == "__main__":
    main()
