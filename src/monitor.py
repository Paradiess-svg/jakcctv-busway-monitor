"""monitor.py -- Tahap 5: monitoring kendaraan TJ vs non-TJ di jalur busway dari video / stream CCTV.

Bukan sistem tilang: hanya mendeteksi, mengklasifikasi, dan mencatat kendaraan yang berada di ROI busway.

Alur per frame (diproses proses_fps kali per detik video, sisanya dilewati supaya kuat di CPU):
  YOLO11 (model v2) + ByteTrack -> tiap kendaraan punya track_id
  -> titik bawah-tengah kotak (roda di jalan) dicek masuk poligon ROI (config/roi/<kamera>.json)
  -> kalau total waktu di ROI >= min_detik_di_roi, kendaraan dicatat SEKALI per track_id
  -> kelas akhir = kelas yang paling sering muncul selama di-track (supaya tidak goyah per frame)
  -> dicatat ke log_tj.csv (tj_bus, mikrotrans) atau log_non_tj.csv (bus_lain, car, motorcycle, truck);
     kend_prioritas diabaikan. Semua aturan ada di config.yaml bagian 'monitor'.

Keluaran <output_dir>/<kamera>/<nama_video>/:
  log_tj.csv, log_non_tj.csv, snapshot/*.jpg, ringkasan.txt, hasil.mp4 (kalau --video-out)

Contoh:
  python src/monitor.py --help
  python src/monitor.py --video data/clips/c01_taman_literasi/2026-10-25/075057.mp4 --video-out
  python src/monitor.py --video data/clips/c04_gerbang_pemuda/2026-10-25/190000.mp4 --detik 60
Kelas Monitor dipakai juga oleh dasbor (Tahap 6) untuk mode upload dan mode live.
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
WARNA = {"tj": (60, 180, 60), "non_tj": (40, 40, 230), "abaikan": (200, 200, 200), "luar": (160, 160, 160)}
KOLOM = ["waktu", "detik_video", "kamera", "track_id", "kelas", "kelompok", "conf_rata2", "detik_di_roi", "snapshot",
         "box", "diam"]


def ioa(a, b):
    """Bagian kotak a yang tertutup kotak b (0-1). Untuk kotak bagian dari kendaraan diam yang sama."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    luas = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / luas if luas > 0 else 0.0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def muat_config(path=ROOT / "config" / "config.yaml"):
    return yaml.safe_load(open(path, encoding="utf-8"))["monitor"]


def kamera_dari_path(p: Path):
    """data/clips/<kamera>/<tanggal>/<HHMMSS>.mp4 -> (kamera, waktu mulai) kalau polanya cocok."""
    try:
        return p.parent.parent.name, datetime.strptime(f"{p.parent.name} {p.stem}", "%Y-%m-%d %H%M%S")
    except ValueError:
        return None, None


class Monitor:
    """Satu kamera. Panggil olah(frame, detik) berulang; hasilnya frame beranotasi + kejadian baru."""

    def __init__(self, kamera, cfg=None, model=None, out_dir=None, waktu_mulai=None):
        from ultralytics import YOLO
        self.cfg = cfg or muat_config()
        self.cfg_penuh = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
        self.kamera = kamera
        self.model = model or YOLO(str(ROOT / self.cfg["model"]))
        self.nama = self.model.names
        self.grup = {k: g for g, ks in self.cfg["kelompok"].items() for k in ks}
        roi = json.load(open(ROOT / self.cfg["roi_dir"] / f"{kamera}.json", encoding="utf-8"))
        self.roi_ukuran = roi["frame_size"]
        self.roi_asli = [np.array(p, np.float32) for p in roi["polygons"]]
        self.roi = None
        self.waktu_mulai = waktu_mulai
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            (self.out_dir / "snapshot").mkdir(parents=True, exist_ok=True)
            for g in ("tj", "non_tj"):
                with open(self.out_dir / f"log_{g}.csv", "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(KOLOM)
        self.track = defaultdict(lambda: {"kelas": Counter(), "conf": [], "di_roi": 0.0, "dicatat": False,
                                          "box_awal": None, "terakhir": 0.0})
        self.detik_lalu = None
        self.hitung = Counter()   # jumlah kendaraan tercatat per kelompok
        self.kejadian = []        # semua kejadian (dict)
        self.posisi_tercatat = [] # (detik, box, grup, diam) untuk menyaring kendaraan diam yang ID-nya berganti
        self.dobel = 0
        self.lampu = "kuning"     # status ROI frame terakhir: merah = ada non-TJ, hijau = ada TJ, kuning = kosong
        self.jumlah_kendaraan = 0 # kendaraan terdeteksi di frame terakhir (seluruh frame), untuk kepadatan
        self._waktu = None

    def _siapkan_roi(self, w, h):
        sx, sy = w / self.roi_ukuran[0], h / self.roi_ukuran[1]
        self.roi = [(p * [sx, sy]).astype(np.int32) for p in self.roi_asli]

    def di_roi(self, x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in self.roi)

    def olah(self, frame, detik, waktu=None):
        """frame BGR, detik = posisi waktu (detik) di video/stream, waktu = jam nyata (datetime) kalau live.
        Return (frame_anotasi, kejadian_baru)."""
        self._waktu = waktu
        h, w = frame.shape[:2]
        if self.roi is None:
            self._siapkan_roi(w, h)
        if self.cfg.get("preprocessing"):
            # sama persis dengan buat_dataset_clahe.py (aturan #4); siang/malam dari jam rekaman
            from preprocessing import process
            waktu = (self.waktu_mulai + timedelta(seconds=detik)) if self.waktu_mulai else datetime.now()
            frame, _ = process(frame, self.cfg_penuh, waktu.hour + waktu.minute / 60)
        dt = 0.0 if self.detik_lalu is None else min(detik - self.detik_lalu, 2.0)
        self.detik_lalu = detik
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.cfg["imgsz"],
                             conf=self.cfg["conf"], verbose=False)[0]
        tampil = frame.copy()
        cv2.polylines(tampil, self.roi, True, (0, 255, 255), 2)
        baru = []
        ada = set()
        self.jumlah_kendaraan = 0 if r.boxes is None else len(r.boxes)
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                    r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                t = self.track[tid]
                t["kelas"][self.nama[c]] += 1
                t["conf"].append(cf)
                kelas = t["kelas"].most_common(1)[0][0]
                grup = self.grup.get(kelas, "abaikan")
                masuk = self.di_roi((x1 + x2) / 2, y2)
                t["terakhir"] = detik
                if masuk:
                    t["di_roi"] += dt
                    if t["box_awal"] is None:
                        t["box_awal"] = (x1, y1, x2, y2)
                    if grup != "abaikan":
                        ada.add(grup)
                if masuk and not t["dicatat"] and grup != "abaikan" and t["di_roi"] >= self.cfg["min_detik_di_roi"]:
                    t["dicatat"] = True
                    if self._sudah_tercatat((x1, y1, x2, y2), grup, detik):
                        self.dobel += 1
                    else:
                        baru.append(self._catat(frame, tid, kelas, grup, t, detik, (x1, y1, x2, y2)))
                warna = WARNA[grup] if masuk else WARNA["luar"]
                cv2.rectangle(tampil, (int(x1), int(y1)), (int(x2), int(y2)), warna, 2)
                teks = f"#{tid} {kelas}" + (" *" if t["dicatat"] else "")
                cv2.putText(tampil, teks, (int(x1), max(int(y1) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1,
                            cv2.LINE_AA)
        self.lampu = "merah" if "non_tj" in ada else "hijau" if "tj" in ada else "kuning"
        self._buang_track_lama(detik)
        cv2.rectangle(tampil, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(tampil, f"{self.kamera}  TJ: {self.hitung['tj']}  non-TJ: {self.hitung['non_tj']}",
                    (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return tampil, baru

    def _buang_track_lama(self, detik, batas=60.0):
        """Hemat RAM saat live berjam-jam: track yang sudah > batas detik tidak terlihat dibuang."""
        if len(self.track) > 200:
            for tid in [k for k, t in self.track.items() if detik - t["terakhir"] > batas]:
                del self.track[tid]
        jendela = max(self.cfg["dobel_detik"], self.cfg.get("diam_detik", 600))
        self.posisi_tercatat = [p for p in self.posisi_tercatat if detik - p[0] <= jendela]
        if len(self.kejadian) > 500:
            self.kejadian = self.kejadian[-500:]

    def _sudah_tercatat(self, box, grup, detik):
        """Dobel kalau (1) posisi hampir sama (IoU) dalam dobel_detik, atau (2) kotak baru sebagian besar berada di
        dalam kotak kendaraan DIAM (parkir/ngetem) yang sudah tercatat dalam diam_detik -- bus besar yang diam
        sering dapat ID baru dengan kotak hanya sebagian badan bus."""
        for d, b, g, diam in self.posisi_tercatat:
            if g != grup:
                continue
            if detik - d <= self.cfg["dobel_detik"] and iou(b, box) > self.cfg["dobel_iou"]:
                return True
            if diam and detik - d <= self.cfg.get("diam_detik", 600) and                     max(ioa(box, b), ioa(b, box)) > self.cfg.get("diam_ioa", 0.6):
                return True
        return False

    def _catat(self, frame, tid, kelas, grup, t, detik, box):
        self.hitung[grup] += 1
        # diam = pusat kotak bergeser < 15% lebar kotak sejak pertama masuk ROI
        ba = t["box_awal"] or box
        geser = abs((ba[0] + ba[2]) / 2 - (box[0] + box[2]) / 2) + abs((ba[1] + ba[3]) / 2 - (box[1] + box[3]) / 2)
        diam = geser < 0.15 * max(box[2] - box[0], 1)
        self.posisi_tercatat.append((detik, box, grup, diam))
        # live: jam nyata dari pemanggil (waktu stream bisa melambat saat stream putus-sambung)
        waktu = self._waktu or ((self.waktu_mulai + timedelta(seconds=detik)) if self.waktu_mulai else datetime.now())
        k = {"waktu": waktu.strftime("%Y-%m-%d %H:%M:%S"), "detik_video": round(detik, 1), "kamera": self.kamera,
             "track_id": tid, "kelas": kelas, "kelompok": grup, "conf_rata2": round(float(np.mean(t["conf"])), 3),
             "detik_di_roi": round(t["di_roi"], 1), "snapshot": "", "box": " ".join(str(int(v)) for v in box),
             "diam": int(diam)}
        if self.out_dir:
            snap = frame.copy()
            x1, y1, x2, y2 = map(int, box)
            cv2.polylines(snap, self.roi, True, (0, 255, 255), 2)
            cv2.rectangle(snap, (x1, y1), (x2, y2), WARNA[grup], 2)
            cv2.putText(snap, f"{kelas} {k['waktu']}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        WARNA[grup], 1, cv2.LINE_AA)
            nama = f"{waktu:%Y%m%d_%H%M%S}_{grup}_{kelas}_id{tid}.jpg"
            cv2.imwrite(str(self.out_dir / "snapshot" / nama), snap)
            k["snapshot"] = f"snapshot/{nama}"
            with open(self.out_dir / f"log_{grup}.csv", "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, KOLOM).writerow(k)
        self.kejadian.append(k)
        return k


def olah_video(video, kamera=None, out=None, detik_maks=None, video_out=False, cfg=None, model=None, progres=None):
    """Proses satu file video dari awal sampai akhir. Dipakai CLI dan mode upload dasbor."""
    cfg = cfg or muat_config()
    video = Path(video)
    kam_path, mulai = kamera_dari_path(video)
    kamera = kamera or kam_path
    if not kamera:
        raise SystemExit("Nama kamera tidak bisa ditebak dari path; isi --kamera")
    out = Path(out) if out else ROOT / cfg["output_dir"] / kamera / video.stem
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lompat = max(1, round(fps / cfg["proses_fps"]))
    mon = Monitor(kamera, cfg, model, out, mulai)
    vw = None
    t0, n, i = time.time(), 0, 0
    while True:
        ok, frame = cap.read() if i % lompat == 0 else (cap.grab(), None)
        if not ok:
            break
        detik = i / fps
        if detik_maks and detik > detik_maks:
            break
        if frame is not None:
            tampil, _ = mon.olah(frame, detik)
            n += 1
            if video_out:
                if vw is None:
                    vw = cv2.VideoWriter(str(out / "hasil.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cfg["proses_fps"],
                                         (tampil.shape[1], tampil.shape[0]))
                vw.write(tampil)
            if progres:
                progres(i / max(total, 1), tampil, mon)
        i += 1
    cap.release()
    if vw:
        vw.release()
    durasi = time.time() - t0
    ringkas = (f"Video: {video}\nKamera: {kamera}\nDurasi video diproses: {i / fps:.0f} s, {n} frame dianalisis "
               f"({cfg['proses_fps']} fps), waktu proses {durasi:.0f} s ({1000 * durasi / max(n, 1):.0f} ms/frame)\n"
               f"Tercatat di ROI >= {cfg['min_detik_di_roi']} s: TJ {mon.hitung['tj']}, non-TJ {mon.hitung['non_tj']} "
               f"(disaring sbg kendaraan diam yang sama: {mon.dobel})\n"
               + "".join(f"  {k}: {v}\n" for k, v in Counter(e['kelas'] for e in mon.kejadian).most_common()))
    (out / "ringkasan.txt").write_text(ringkas, encoding="utf-8")
    return mon, ringkas, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="file video (mis. data/clips/<kamera>/<tanggal>/<HHMMSS>.mp4)")
    ap.add_argument("--kamera", help="ID kamera (default ditebak dari path data/clips/<kamera>/...)")
    ap.add_argument("--out", help="folder keluaran (default outputs/monitor/<kamera>/<nama_video>)")
    ap.add_argument("--detik", type=float, help="hanya proses N detik pertama (untuk uji cepat)")
    ap.add_argument("--video-out", action="store_true", help="simpan video beranotasi hasil.mp4")
    a = ap.parse_args()
    _, ringkas, out = olah_video(ROOT / a.video if not Path(a.video).is_absolute() else a.video, a.kamera, a.out,
                                 a.detik, a.video_out)
    print(ringkas + f"Keluaran: {out}")


if __name__ == "__main__":
    main()
