"""Cek cakupan capture: berapa frame per kamera, per tanggal, per jam, dan apakah harinya "lengkap".

Dipakai untuk memutuskan sebuah hari capture sudah cukup atau belum (BUKAN jumlah frame total).
Sebuah hari dianggap LENGKAP kalau minimal 3 dari 4 periode terisi (>= --min-per-periode frame):
  pagi 05-09 | siang 10-14 | sore 15-18 | malam 19-23

Contoh:
  python src/cek_cakupan.py                      # semua kamera di data/raw
  python src/cek_cakupan.py --kamera c13_gatsu   # satu kamera
  python src/cek_cakupan.py --min-per-periode 300

Struktur yang dibaca: data/raw/<kamera>/<YYYY-MM-DD>/<HHMMSS>[_suffix].jpg (jam diambil dari nama file).
Hanya membaca, tidak mengubah atau menghapus data.
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

PERIODE = {"pagi": range(5, 10), "siang": range(10, 15), "sore": range(15, 19), "malam": range(19, 24)}
JAM_RE = re.compile(r"^(\d{2})\d{4}")


def periode_dari_jam(jam):
    for nama, rentang in PERIODE.items():
        if jam in rentang:
            return nama
    return None  # 00-04: di luar jam capture normal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw", help="folder data mentah (default data/raw)")
    ap.add_argument("--kamera", help="batasi ke satu kamera, mis. c13_gatsu")
    ap.add_argument("--min-per-periode", type=int, default=200,
                    help="frame minimal agar sebuah periode dianggap terisi (default 200, ±17 menit pada selang 5 dtk)")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    if not raw.is_dir():
        raise SystemExit(f"Folder tidak ada: {raw} (jalankan dari folder Project-TJ)")

    kameras = sorted(p for p in raw.iterdir() if p.is_dir() and (not args.kamera or p.name == args.kamera))
    if not kameras:
        raise SystemExit("Tidak ada folder kamera yang cocok.")

    for kam in kameras:
        print(f"\n=== {kam.name} ===")
        per_hari = defaultdict(Counter)
        for tgl in sorted(p for p in kam.iterdir() if p.is_dir()):
            for f in tgl.glob("*.jpg"):
                m = JAM_RE.match(f.name)
                if m:
                    per_hari[tgl.name][int(m.group(1))] += 1
        if not per_hari:
            print("  (belum ada frame)")
            continue
        for tgl, jam_count in sorted(per_hari.items()):
            per_periode = Counter()
            for jam, n in jam_count.items():
                p = periode_dari_jam(jam)
                if p:
                    per_periode[p] += n
            terisi = [p for p in PERIODE if per_periode[p] >= args.min_per_periode]
            kosong = [p for p in PERIODE if p not in terisi]
            status = "LENGKAP" if len(terisi) >= 3 else "BELUM LENGKAP"
            print(f"  {tgl}  total {sum(jam_count.values())} frame  ->  {status}"
                  f"  (terisi: {', '.join(terisi) or '-'}; kurang: {', '.join(kosong) or '-'})")
            print("    per jam: " + ", ".join(f"{j:02d}h={jam_count[j]}" for j in sorted(jam_count)))


if __name__ == "__main__":
    main()
