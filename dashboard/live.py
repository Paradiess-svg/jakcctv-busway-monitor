"""live.py -- mode live dasbor: stream HLS JakCCTV -> Monitor (YOLO + ByteTrack + ROI) per kamera.

Struktur:
  PembacaStream (1 thread + 1 ffmpeg per kamera): decode HLS -> frame BGR 640x360, antrean pendek frame
  LiveManager   (1 thread inference): bergiliran ke kamera yang punya frame baru -> Monitor.olah()
                -> JPEG beranotasi terbaru per kamera + kejadian (log TJ / non-TJ) + snapshot.
Satu thread inference saja karena CPU laptop (4 core): 4 thread YOLO bersamaan malah saling rebut CPU.
Tiap kamera punya instance YOLO sendiri, karena state ByteTrack (persist=True) melekat di model.

Keluaran sesi: outputs/monitor/live/<kamera>/<YYYY-MM-DD_HHMMSS>/ (log_tj.csv, log_non_tj.csv, snapshot/).
Tidak merekam video (disk laptop terbatas); yang disimpan hanya log + snapshot kendaraan tercatat.
Pengaman RAM (19 Nov, live sempat jalan berhari-hari -> pagefile bengkak): live berhenti sendiri kalau dasbor
tidak dibuka selama dasbor.live_idle_menit; model & track dibersihkan tiap stop.
Tiap kamera bisa dimatikan / dinyalakan (= sambung ulang) sendiri. Di GPU (Kaggle/Colab) dibaca live_fps_baca_gpu.
"""
import gc
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from capture import FFMPEG_INPUT, configure_hls_opts, find_ffmpeg, popen_flags  # noqa: E402
from monitor import Monitor, muat_config  # noqa: E402

W, H = 640, 360  # ukuran ROI dan data latih


class PembacaStream(threading.Thread):
    """ffmpeg -> rawvideo bgr24 lewat pipe. Reconnect otomatis dengan backoff."""

    def __init__(self, kamera, url, ffmpeg, fps_baca, antrean_maks=8):
        super().__init__(name=f"baca-{kamera}", daemon=True)
        self.kamera, self.url, self.ffmpeg, self.fps_baca = kamera, url, ffmpeg, fps_baca
        self.stop_evt = threading.Event()
        self.kunci = threading.Lock()
        # HLS datang per segmen (beberapa detik sekaligus), jadi frame ditampung di antrean pendek, bukan
        # hanya frame terbaru. Kalau inference tertinggal, frame paling lama dibuang (maxlen).
        self.antrean = deque(maxlen=antrean_maks)
        self.no_frame, self.waktu_frame, self.dibuang = 0, 0.0, 0
        self.status, self.error = "menyambung", ""
        self.proc = None

    def run(self):
        jeda = 2
        while not self.stop_evt.is_set():
            cmd = [self.ffmpeg, *FFMPEG_INPUT, "-i", self.url, "-an",
                   "-vf", f"fps={self.fps_baca},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
            self.status = "menyambung"
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_flags())
                threading.Thread(target=self._kuras_err, args=(self.proc,), daemon=True).start()
                n = W * H * 3
                while not self.stop_evt.is_set():
                    buf = self.proc.stdout.read(n)
                    if len(buf) < n:
                        break
                    with self.kunci:
                        if len(self.antrean) == self.antrean.maxlen:
                            self.dibuang += 1
                        self.no_frame += 1
                        # detik = waktu di stream (no frame / fps), dipakai aturan >= 2 detik di ROI
                        self.antrean.append((np.frombuffer(buf, np.uint8).reshape(H, W, 3), self.no_frame / self.fps_baca))
                        self.waktu_frame = time.time()
                    self.status, jeda = "hidup", 2
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
            finally:
                self._matikan()
            if self.stop_evt.is_set():
                break
            self.status = "putus"
            self.stop_evt.wait(jeda)
            jeda = min(jeda * 2, 30)
        self.status = "berhenti"

    def _kuras_err(self, proc):
        for baris in proc.stderr:
            t = baris.decode("utf-8", "replace").strip()
            if t:
                self.error = t[-300:]

    def _matikan(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(5)
            except Exception:  # noqa: BLE001
                pass

    def ambil(self):
        """Frame tertua di antrean (frame, detik_stream) atau None."""
        with self.kunci:
            return self.antrean.popleft() if self.antrean else None

    def info(self):
        with self.kunci:
            return self.no_frame, self.waktu_frame, len(self.antrean), self.dibuang

    def stop(self):
        self.stop_evt.set()
        self._matikan()




def ada_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class LiveManager:
    """Grid live maks 4 kamera. Tiap kamera bisa dimatikan/dinyalakan sendiri (nyalakan = sambung ulang/reset).
    Kamera yang dimatikan tidak dianalisis dan tidak mencatat apa pun sampai dinyalakan lagi."""

    def __init__(self):
        self.cfg_penuh = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
        self.kamera_cfg = self.cfg_penuh["cameras"]
        self.dasbor = self.cfg_penuh.get("dasbor", {})
        self.gpu = ada_gpu()
        self.ffmpeg = None
        self.kunci = threading.RLock()
        self.daftar = []                     # kamera di grid live (urutan tampil)
        self.pembaca, self.monitor, self.jpeg, self.info, self.kepadatan = {}, {}, {}, {}, {}
        self.kejadian = deque(maxlen=2000)
        self.no_kejadian = 0
        self.thread, self.stop_evt = None, threading.Event()
        self.mulai = None
        self.disentuh = time.time()
        threading.Thread(target=self._penjaga_idle, name="penjaga-idle", daemon=True).start()

    # ---------------- penjaga RAM ----------------
    def sentuh(self):
        """Dipanggil tiap ada permintaan dari dasbor; kalau lama tidak ada, live dihentikan."""
        self.disentuh = time.time()

    def _penjaga_idle(self):
        while True:
            time.sleep(30)
            batas = 60 * self.dasbor.get("live_idle_menit", 10)
            if self.pembaca and time.time() - self.disentuh > batas:
                print(f"[live] dasbor tidak dibuka > {batas // 60} menit -> live dihentikan", flush=True)
                self.stop()

    # ---------------- kontrol ----------------
    def _cek(self, k):
        if k not in self.kamera_cfg:
            raise ValueError(f"kamera tidak dikenal: {k}")
        if not (ROOT / muat_config()["roi_dir"] / f"{k}.json").exists():
            raise ValueError(f"ROI belum ada untuk {k}")

    def _fps_baca(self, n):
        cfg = muat_config()
        if self.gpu:
            return cfg.get("live_fps_baca_gpu", 5)
        # CPU: 1-2 kamera 2 fps (setara 5 fps pada uji hitungan); 3-4 kamera ~1 fps (batas CPU laptop)
        return 2 if n <= 2 else cfg.get("live_fps_baca", 1)

    def start(self, kamera_list):
        """Ganti seluruh grid: semua kamera di daftar dinyalakan."""
        for k in kamera_list:
            self._cek(k)
        self.stop()
        self.daftar = list(kamera_list)
        self.mulai = datetime.now()
        for k in self.daftar:
            self.nyalakan(k)
            time.sleep(0.5)  # jangan menyambung semua kamera di detik yang sama

    def nyalakan(self, k):
        """Nyalakan / sambung ulang (reset) satu kamera. Sesi log baru dibuat."""
        self._cek(k)
        self.matikan(k)
        if self.ffmpeg is None:
            self.ffmpeg = find_ffmpeg(self.cfg_penuh)
            configure_hls_opts(self.ffmpeg)
        if k not in self.daftar:
            if len(self.daftar) >= 4:
                raise ValueError("grid live maks 4 kamera")
            self.daftar.append(k)
        cfg = muat_config()
        mulai = datetime.now()
        self.mulai = self.mulai or mulai
        out = ROOT / cfg["output_dir"] / "live" / k / mulai.strftime("%Y-%m-%d_%H%M%S")
        mon = Monitor(k, cfg, out_dir=out, waktu_mulai=mulai)
        fps = self._fps_baca(len(self.pembaca) + 1)
        p = PembacaStream(k, self.kamera_cfg[k]["url"], self.ffmpeg, fps, antrean_maks=max(8, int(fps * 8)))
        with self.kunci:
            self.monitor[k] = mon
            self.kepadatan[k] = deque(maxlen=20)
            self.info[k] = {"kamera": k, "fps_analisis": 0.0, "ms_per_frame": 0, "frame_dianalisis": 0,
                            "folder": out.relative_to(ROOT).as_posix(),
                            "url_folder": "/files/" + out.relative_to(ROOT / cfg["output_dir"]).as_posix(),
                            "dinyalakan": mulai.strftime("%H:%M:%S"), "_t": deque(maxlen=10)}
            self.pembaca[k] = p
        p.start()
        if self.thread is None or not self.thread.is_alive():
            self.stop_evt = threading.Event()
            self.thread = threading.Thread(target=self._loop, name="inference", daemon=True)
            self.thread.start()

    def matikan(self, k):
        """Putus koneksi kamera k; pencatatan kamera ini berhenti sampai dinyalakan lagi."""
        with self.kunci:
            p = self.pembaca.pop(k, None)
            self.monitor.pop(k, None)
            self.jpeg.pop(k, None)
            self.kepadatan.pop(k, None)
        if p:
            p.stop()
            gc.collect()

    def stop(self):
        """Matikan semua (daftar grid tetap diingat)."""
        for k in list(self.pembaca):
            self.matikan(k)
        self.stop_evt.set()
        if self.thread:
            self.thread.join(10)
        self.thread = None
        gc.collect()  # lepas model YOLO + state ByteTrack

    # ---------------- loop inference ----------------
    def _loop(self):
        i = 0
        while not self.stop_evt.is_set():
            with self.kunci:
                urutan = list(self.pembaca)
            kerja = False
            for _ in range(len(urutan)):
                k = urutan[i % len(urutan)]
                i += 1
                with self.kunci:
                    p, mon, inf = self.pembaca.get(k), self.monitor.get(k), self.info.get(k)
                if p is None or mon is None:
                    continue
                item = p.ambil()
                if item is None:
                    continue
                frame, detik = item
                t0 = time.time()
                try:
                    tampil, baru = mon.olah(frame.copy(), detik, datetime.now())
                except Exception as e:  # noqa: BLE001
                    inf["error"] = str(e)
                    continue
                ok, jpg = cv2.imencode(".jpg", tampil, [cv2.IMWRITE_JPEG_QUALITY, 75])
                inf["_t"].append(time.time())
                inf["frame_dianalisis"] += 1
                inf["ms_per_frame"] = round(1000 * (time.time() - t0))
                ts = inf["_t"]
                inf["fps_analisis"] = round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
                with self.kunci:
                    if k not in self.pembaca:   # dimatikan saat frame ini diproses
                        continue
                    self.kepadatan[k].append(mon.jumlah_kendaraan)
                    if ok:
                        self.jpeg[k] = jpg.tobytes()
                    for e in baru:
                        self.no_kejadian += 1
                        self.kejadian.append({"id": self.no_kejadian, **e,
                                              "snapshot_url": f"{inf['url_folder']}/{e['snapshot']}" if e.get("snapshot") else ""})
                kerja = True
                break  # satu frame per putaran, lalu kamera berikutnya (adil)
            if not kerja:
                self.stop_evt.wait(0.03)

    # ---------------- baca status ----------------
    def status(self):
        kam = []
        with self.kunci:
            for k in self.daftar:
                p, m, inf = self.pembaca.get(k), self.monitor.get(k), self.info.get(k, {})
                if p is None or m is None:
                    kam.append({"kamera": k, "nyala": False, "stream": "dimatikan", "lampu": "mati",
                                "hitung": {"tj": 0, "non_tj": 0}, "fps_analisis": 0, "ms_per_frame": 0,
                                "kepadatan": None, "kendaraan_per_frame": None})
                    continue
                no, wf, antre, buang = p.info()
                kam.append({**{a: b for a, b in inf.items() if not a.startswith("_")}, "nyala": True,
                            "stream": p.status, "frame_masuk": no, "antrean": antre, "frame_dibuang": buang,
                            "detik_sejak_frame": round(time.time() - wf, 1) if wf else None,
                            "error_stream": p.error if p.status != "hidup" else "",
                            "hitung": {"tj": m.hitung["tj"], "non_tj": m.hitung["non_tj"]}, "disaring_dobel": m.dobel,
                            "lampu": m.lampu if p.status == "hidup" else "mati", **self._kepadatan(k)})
        return {"aktif": bool(self.pembaca), "gpu": self.gpu,
                "mulai": self.mulai.strftime("%Y-%m-%d %H:%M:%S") if self.mulai else None, "kamera": kam}

    def _kepadatan(self, k):
        """Perkiraan kasar dari rata-rata jumlah kendaraan terdeteksi per frame (seluruh frame, bukan hanya ROI)."""
        q = self.kepadatan.get(k)
        if not q:
            return {"kendaraan_per_frame": None, "kepadatan": None}
        r = sum(q) / len(q)
        return {"kendaraan_per_frame": round(r, 1), "kepadatan": "lancar" if r < 8 else "sedang" if r < 16 else "padat"}

    def jpeg_terbaru(self, k):
        with self.kunci:
            return self.jpeg.get(k)

    def kejadian_sejak(self, setelah_id=0, kelompok=None, kamera=None, batas=200):
        with self.kunci:
            hasil = [e for e in self.kejadian if e["id"] > setelah_id
                     and (not kelompok or e["kelompok"] == kelompok) and (not kamera or e["kamera"] == kamera)]
        return hasil[-batas:]
