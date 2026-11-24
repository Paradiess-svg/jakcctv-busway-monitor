"""pilih_frame.py -- Tugas 3.4: pilih frame beragam dari data mentah untuk dataset v1.

HANYA MEMBACA data/raw (tidak menyalin, mengubah, atau menghapus gambar). Keluaran:
  outputs/dataset_v1_frames.csv     daftar frame terpilih (path, kamera, tanggal, jam, periode, split, ...)
  outputs/dataset_v1_ringkasan.csv  ringkasan jumlah per kamera x split x periode

Cara kerja:
  1. Baca split dari config/split_v1.json (per kamera-tanggal, bukan acak per frame).
  2. Kandidat = frame dengan jarak waktu >= min_gap_s (membuang tetangga yang nyaris kembar).
  3. Tiap kandidat diberi skor gerakan (porsi piksel berubah vs frame berikutnya, ~5 dtk) + kecerahan.
  4. Kuota per kamera/split dibagi ke strata (tanggal x periode) sesuai bobot periode (malam dibobot besar).
  5. Di tiap strata: ~porsi_idle dari frame sepi (contoh negatif), sisanya frame ramai; dipilih dengan
     farthest-point sampling pada thumbnail supaya adegannya beragam.
Catatan: gerakan adalah PENDEKATAN untuk "ada kendaraan"; ROI busway belum dipakai.

Contoh:
  python src/pilih_frame.py --help
  python src/pilih_frame.py --kamera c01_taman_literasi      # coba satu kamera dulu
  python src/pilih_frame.py                                   # semua kamera di config
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PERIODE = [("pagi", 5, 9), ("siang", 10, 14), ("sore", 15, 18), ("malam", 19, 23)]


def periode_of(h):
    for nama, a, b in PERIODE:
        if a <= h <= b:
            return nama
    return "malam" if h >= 19 or h < 5 else "pagi"


def list_frames(raw_dirs, kamera, tanggal):
    """Semua jpg satu kamera-tanggal, terurut waktu: [(detik_hari, Path)]."""
    out = []
    for rd in raw_dirs:
        d = (ROOT / rd / kamera / tanggal)
        if d.is_dir():
            for f in d.glob("*.jpg"):
                s = f.stem[:6]
                if len(s) == 6 and s.isdigit():
                    out.append((int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:]), f))
    out.sort(key=lambda x: (x[0], x[1].name))
    return out


def baca_kecil(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        return None
    return cv2.resize(img, (80, 45), interpolation=cv2.INTER_AREA)


def hitung_fitur(item):
    """item = (detik, path, path_tetangga) -> dict fitur atau None kalau rusak."""
    t, p, q = item
    a = baca_kecil(p)
    if a is None:
        return None
    b = baca_kecil(q) if q is not None else None
    gerak = 0.0
    if b is not None:
        gerak = float((cv2.absdiff(a, b) > 25).mean())
    thumb = cv2.resize(a, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    return {"t": t, "path": p, "gerak": gerak, "cerah": float(a.mean()), "thumb": thumb}


def kandidat(frames, min_gap):
    """Ambil frame dengan jarak >= min_gap detik; tetangga = frame berikutnya (atau sebelumnya)."""
    pilih, last = [], -10**9
    for i, (t, p) in enumerate(frames):
        if t - last >= min_gap:
            if i + 1 < len(frames) and frames[i + 1][0] - t <= 15:
                q = frames[i + 1][1]
            elif i > 0 and t - frames[i - 1][0] <= 15:
                q = frames[i - 1][1]
            else:
                q = None
            pilih.append((t, p, q))
            last = t
    return pilih


def alokasi(kuota, kapasitas, bobot):
    """Bagi kuota ke strata proporsional bobot, dibatasi kapasitas (water-filling)."""
    hasil = {k: 0 for k in kapasitas}
    aktif = {k for k in kapasitas if kapasitas[k] > 0 and bobot[k] > 0}
    sisa = kuota
    while sisa > 0 and aktif:
        tb = sum(bobot[k] for k in aktif)
        diberi = 0
        for k in sorted(aktif):
            jatah = int(sisa * bobot[k] / tb)
            jatah = min(jatah, kapasitas[k] - hasil[k])
            hasil[k] += jatah
            diberi += jatah
        sisa -= diberi
        aktif = {k for k in aktif if hasil[k] < kapasitas[k]}
        if diberi == 0:  # sisa pembulatan: bagi satu-satu
            for k in sorted(aktif, key=lambda k: -bobot[k]):
                if sisa <= 0:
                    break
                hasil[k] += 1
                sisa -= 1
            aktif = {k for k in aktif if hasil[k] < kapasitas[k]}
    return hasil


def fps(items, n, rng):
    """Farthest-point sampling pada thumbnail; mulai dari item dengan gerakan terbesar."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    X = np.stack([it["thumb"] for it in items])
    awal = int(np.argmax([it["gerak"] for it in items]))
    terpilih = [awal]
    d = np.linalg.norm(X - X[awal], axis=1)
    while len(terpilih) < n:
        j = int(np.argmax(d))
        terpilih.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return [items[i] for i in terpilih]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/split_v1.json")
    ap.add_argument("--kamera", help="proses satu kamera saja (default: semua di config)")
    ap.add_argument("--out", default="outputs/dataset_v1_frames.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    pw = cfg["bobot_periode"]
    mulai_malam = cfg["mulai_malam_jam"]  # kamera auto-exposure -> kecerahan tidak bisa dipakai; 18:01 sudah gelap
    # sanity: satu tanggal tidak boleh di dua split
    for cam, sp in cfg["kamera"].items():
        semua = [d for k in ("train", "valid", "test", "cadangan") for d in sp.get(k, [])]
        assert len(semua) == len(set(semua)), f"{cam}: ada tanggal di lebih dari satu split"

    baris = []
    for cam, sp in cfg["kamera"].items():
        if args.kamera and cam != args.kamera:
            continue
        for split in ("train", "valid", "test"):
            strata = {}  # (tanggal, periode) -> list fitur
            for tgl in sp.get(split, []):
                frames = list_frames(cfg["raw_dirs"], cam, tgl)
                if not frames:
                    print(f"[!] {cam} {tgl}: tidak ada frame (dilewati)")
                    continue
                kand = kandidat(frames, cfg["min_gap_s"])
                with ThreadPoolExecutor(args.threads) as ex:
                    fitur = [f for f in ex.map(hitung_fitur, kand) if f is not None]
                for f in fitur:
                    if f["cerah"] < 8:      # hampir hitam -> buang
                        continue
                    f["tgl"] = tgl
                    strata.setdefault((tgl, periode_of(f["t"] // 3600)), []).append(f)
                print(f"{cam} {split} {tgl}: {len(frames)} frame -> {len(kand)} kandidat")
            if not strata:
                continue
            n_tgl_per = defaultdict(int)
            for (tgl, per) in strata:
                n_tgl_per[per] += 1
            bobot = {k: pw[k[1]] / n_tgl_per[k[1]] for k in strata}
            kap = {k: len(v) for k, v in strata.items()}
            jatah = alokasi(cfg["quota_per_kamera"][split], kap, bobot)
            for k, n in jatah.items():
                items = sorted(strata[k], key=lambda f: f["gerak"])
                med = len(items) // 2
                sepi, ramai = items[:med], items[med:]
                n_idle = min(int(round(n * cfg["porsi_idle"])), len(sepi))
                pilih = [(x, True) for x in fps(sepi, n_idle, rng)]
                pilih += [(x, False) for x in fps(ramai, n - n_idle, rng)]
                for x, idle in pilih:
                    t = x["t"]
                    baris.append({
                        "path": str(x["path"].relative_to(ROOT)).replace("\\", "/"),
                        "kamera": cam, "tanggal": k[0], "jam": f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}",
                        "periode": k[1], "split": split,
                        "siang_malam": "malam" if (t >= mulai_malam * 3600 or t < 5.5 * 3600) else "siang",
                        "kecerahan": round(x["cerah"], 1), "gerak": round(x["gerak"], 4),
                        "idle": int(idle),
                    })
            print(f"== {cam} {split}: terpilih {sum(jatah.values())} (kuota {cfg['quota_per_kamera'][split]})")

    baris.sort(key=lambda r: (r["kamera"], r["split"], r["tanggal"], r["jam"]))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    kolom = ["path", "kamera", "tanggal", "jam", "periode", "split", "siang_malam", "kecerahan", "gerak", "idle"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, kolom)
        w.writeheader()
        w.writerows(baris)

    ring = defaultdict(int)
    for r in baris:
        ring[(r["kamera"], r["split"], r["periode"], r["siang_malam"])] += 1
    with open(out.with_name("dataset_v1_ringkasan.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kamera", "split", "periode", "siang_malam", "jumlah"])
        for k in sorted(ring):
            w.writerow([*k, ring[k]])

    total = len(baris)
    malam = sum(1 for r in baris if r["siang_malam"] == "malam")
    idle = sum(r["idle"] for r in baris)
    print(f"\nTOTAL {total} frame | malam (jam>={mulai_malam}) {malam} ({100*malam/max(total,1):.0f}%) | idle {idle}")
    for s in ("train", "valid", "test"):
        print(f"  {s}: {sum(1 for r in baris if r['split']==s)}")
    print(f"Ditulis: {out}")


if __name__ == "__main__":
    main()
