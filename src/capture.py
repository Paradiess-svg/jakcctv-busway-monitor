#!/usr/bin/env python3
"""
capture.py - Tahap 1: Akuisisi data dari CCTV publik JakCCTV (HLS .m3u8).

Yang dikerjakan:
  1. Satu thread + satu proses ffmpeg per kamera -> frame JPEG disimpan ke
     data/raw/<kamera>/<YYYY-MM-DD>/<HHMMSS>.jpg   (tidak pernah menimpa file lama)
  2. Mode "interval"  : simpan 1 frame tiap capture.frame_interval_s detik.
     Mode "motion"    : simpan hanya saat ada gerakan di ROI busway (config/roi/<kamera>.json);
                        kalau sepi, simpan 1 frame tiap idle_interval_s sebagai contoh negatif.
  3. Klip video pendek (tanpa re-encode) berkala ke data/clips/ untuk uji tracking.
  4. Otomatis: reconnect + backoff, buang frame hitam/duplikat, restart kalau stream beku,
     jendela jam aktif, berhenti kalau disk hampir penuh, log berputar di logs/.

Contoh:
  python src/capture.py --snapshot --node dihar    # tes: ambil 1 frame per kamera lalu selesai
  python src/capture.py --node dihar               # jalan terus (Windows: Task Scheduler)
  python src/capture.py --cameras c13_gatsu        # hanya kamera tertentu
  python src/capture.py --node dihar --max-seconds 120 --no-clips   # uji 2 menit

Semua parameter ada di config/config.yaml (aturan #3).
"""
import argparse
import hashlib
import logging
import logging.handlers
import queue
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yaml

cv2.setNumThreads(1)  # laptop terbatas: jangan rebutan CPU dengan thread lain

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("capture")


# --------------------------------------------------------------------------
# Util: config, jam aktif, ffmpeg
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dir(p: str) -> Path:
    """Path relatif di config dihitung dari folder proyek, bukan dari tempat menjalankan."""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def parse_hours(ranges) -> list:
    """["05:00-23:00", "22:00-02:00"] -> [(300, 1380), (1320, 120)] (menit sejak 00:00)."""
    out = []
    for r in ranges or []:
        a, b = r.split("-")
        ah, am = a.split(":")
        bh, bm = b.split(":")
        out.append((int(ah) * 60 + int(am), int(bh) * 60 + int(bm)))
    return out


def in_hours(now: datetime, ranges: list) -> bool:
    if not ranges:
        return True
    m = now.hour * 60 + now.minute
    for a, b in ranges:
        if a <= b and a <= m < b:
            return True
        if a > b and (m >= a or m < b):  # melewati tengah malam
            return True
    return False


def find_ffmpeg(cfg: dict) -> str:
    custom = (cfg["capture"].get("ffmpeg_path") or "").strip()
    exe = custom or shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg tidak ditemukan. Install dulu (Windows: winget install Gyan.FFmpeg, "
                 "Ubuntu: sudo apt install ffmpeg) lalu buka terminal baru.")
    return exe


def popen_flags() -> dict:
    """Proses ffmpeg dipisah dari Ctrl+C terminal supaya kita yang mengatur penutupannya."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# Server memakai ekstensi segmen .hls.fmp4 yang ditolak ffmpeg secara default, jadi pengecekan
# ekstensi harus dimatikan. Nama opsinya beda antar versi ffmpeg:
#   ffmpeg baru (mis. 9.x di Windows)    : -extension_picky 0   (terbukti di laptop Dihar)
#   ffmpeg lebih lama (mis. 6.1 Ubuntu)  : -allowed_extensions ALL
# configure_hls_opts() memilih yang cocok otomatis saat start.
# -live_start_index -1: mulai dari segmen paling baru. Tanpa ini ffmpeg memutar ulang ~3 segmen
# terakhir sekaligus saat start, sehingga frame datang bergerombol & nama waktunya meleset.
FFMPEG_INPUT = ["-hide_banner", "-loglevel", "error", "-live_start_index", "-1"]


def configure_hls_opts(ffmpeg: str) -> None:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-h", "demuxer=hls"], capture_output=True,
                             timeout=20, **popen_flags())
        text = (out.stdout + out.stderr).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        text = ""
    FFMPEG_INPUT.extend(["-extension_picky", "0"] if "extension_picky" in text
                        else ["-allowed_extensions", "ALL"])


def acquire_lock(name: str):
    """Cegah dua instance untuk node yang sama (aman kalau Task Scheduler memicu ulang tiap 15 menit).
    Memakai socket lokal: otomatis lepas saat proses mati, jalan di Windows & Linux."""
    port = 45000 + (zlib.crc32(name.encode()) % 2000)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        return None
    return s


# --------------------------------------------------------------------------
# Penyaring gerakan di ROI (mode "motion")
# --------------------------------------------------------------------------
def load_roi(cam_id: str):
    """Format config/roi/<kamera>.json:
       {"frame_size": [640, 360], "polygons": [[[x,y],[x,y],...]]}   (atau kunci "polygon" untuk satu poligon)
    """
    import json
    path = ROOT / "config" / "roi" / f"{cam_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        polys = data.get("polygons") or [data["polygon"]]
        size = tuple(data.get("frame_size", (640, 360)))
        return {"polygons": [np.array(p, dtype=np.float32) for p in polys], "frame_size": size}
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] ROI %s tidak bisa dibaca (%s) -> pakai mode interval", cam_id, path.name, e)
        return None


class MotionGate:
    def __init__(self, roi: dict, mcfg: dict):
        self.roi = roi
        self.min_ratio = float(mcfg["min_area_ratio"])
        self.warmup = int(mcfg["warmup_frames"])
        self.mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)
        self.kernel = np.ones((3, 3), np.uint8)
        self.mask = None
        self.area = 1
        self.n = 0

    def _build_mask(self, h, w):
        rw, rh = self.roi["frame_size"]
        mask = np.zeros((h, w), np.uint8)
        for p in self.roi["polygons"]:
            pts = (p * np.array([w / rw, h / rh])).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        self.mask = mask
        self.area = max(1, cv2.countNonZero(mask))

    def has_motion(self, gray: np.ndarray):
        """True/False, atau None selama warmup (belum cukup belajar background)."""
        h, w = gray.shape
        if self.mask is None or self.mask.shape != (h, w):
            self._build_mask(h, w)
        fg = self.mog.apply(cv2.GaussianBlur(gray, (5, 5), 0))
        self.n += 1
        if self.n <= self.warmup:
            return None
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        moving = cv2.countNonZero(cv2.bitwise_and(fg, fg, mask=self.mask))
        return (moving / self.area) >= self.min_ratio


# --------------------------------------------------------------------------
# Worker frame per kamera
# --------------------------------------------------------------------------
class CameraWorker(threading.Thread):
    def __init__(self, cam_id, cam, cfg, ffmpeg, stop, mode_override=None):
        super().__init__(name=f"cam-{cam_id}", daemon=True)
        self.cam_id, self.url = cam_id, cam["url"]
        c = cfg["capture"]
        self.c, self.ffmpeg, self.stop_evt = c, ffmpeg, stop
        self.tz = ZoneInfo(c["timezone"])
        self.hours = parse_hours(c.get("active_hours"))
        self.out_dir = resolve_dir(c["output_dir"]) / cam_id
        self.frame_interval = max(2.0, float(c["frame_interval_s"]))  # aturan #12: jangan agresif
        self.idle_interval = float(c["idle_interval_s"])
        q = c["quality"]
        self.black_below = float(q["black_mean_below"])
        self.frozen_after = float(q["frozen_restart_after_s"])
        rc = c["reconnect"]
        self.stall_timeout = float(rc["stall_timeout_s"])
        self.backoff0, self.backoff_max = float(rc["backoff_start_s"]), float(rc["backoff_max_s"])

        mode = mode_override or c["mode"]
        self.gate = None
        if mode == "motion":
            roi = load_roi(cam_id)
            if roi:
                self.gate = MotionGate(roi, c["motion"])
            else:
                log.warning("[%s] mode motion diminta tapi config/roi/%s.json belum ada -> pakai mode interval",
                            cam_id, cam_id)
        self.mode = "motion" if self.gate else "interval"
        self.sample_period = (1.0 / float(c["motion"]["sample_fps"])) if self.gate else self.frame_interval

        self.stats = dict(saved=0, black=0, dup=0, corrupt=0, restarts=0)
        self.last_saved_t = 0.0
        self.last_hash = None
        self.last_change_t = time.time()

    # ---- jalur file ----
    def _now(self):
        return datetime.now(self.tz)

    def _save(self, jpg: bytes, tag: str = ""):
        now = self._now()
        d = self.out_dir / now.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        base = now.strftime("%H%M%S") + (f"_{tag}" if tag else "")
        for i in range(100):  # 'xb' = gagal kalau file sudah ada -> data mentah tidak pernah tertimpa
            p = d / (f"{base}.jpg" if i == 0 else f"{base}_{i}.jpg")
            try:
                with open(p, "xb") as f:
                    f.write(jpg)
                self.stats["saved"] += 1
                self.last_saved_t = time.time()
                return
            except FileExistsError:
                continue

    # ---- proses 1 frame ----
    def handle_frame(self, jpg: bytes) -> bool:
        """Return False kalau stream sebaiknya di-restart (beku)."""
        gray = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.stats["corrupt"] += 1
            return True
        if float(gray.mean()) < self.black_below:
            self.stats["black"] += 1
            return True

        # Beku: frame sama persis (hash byte) terus-menerus. Stream hidup selalu berubah
        # (noise + jam di overlay), jadi toleransi dinyatakan dalam DETIK, bukan jumlah frame.
        h = hashlib.md5(jpg).digest()
        if h == self.last_hash:
            self.stats["dup"] += 1
            return (time.time() - self.last_change_t) < self.frozen_after
        self.last_hash, self.last_change_t = h, time.time()

        since = time.time() - self.last_saved_t
        if self.gate is None:
            self._save(jpg)
            return True
        moving = self.gate.has_motion(gray)
        if moving is None:
            return True
        if moving and since >= self.frame_interval:
            self._save(jpg, "motion")
        elif not moving and since >= self.idle_interval:
            self._save(jpg, "idle")
        return True

    # ---- satu sesi ffmpeg ----
    def _session(self) -> str:
        fps = f"fps=1/{self.sample_period:g}" if self.gate is None else f"fps={1 / self.sample_period:g}"
        cmd = [self.ffmpeg, *FFMPEG_INPUT, "-i", self.url, "-an", "-vf", fps,
               "-q:v", str(self.c["jpeg_quality"]), "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_flags())
        q: "queue.Queue" = queue.Queue()
        errs: deque = deque(maxlen=5)
        threading.Thread(target=self._drain_err, args=(proc, errs), daemon=True).start()
        threading.Thread(target=self._read_jpegs, args=(proc, q), daemon=True).start()

        start = last = time.time()
        got = 0
        reason = "stop"
        try:
            while not self.stop_evt.is_set():
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - last > self.stall_timeout:
                        reason = "stall"
                        break
                    if not in_hours(self._now(), self.hours):
                        reason = "off_hours"
                        break
                    continue
                if item is None:
                    reason = "eof"
                    break
                last = time.time()
                got += 1
                if not self.handle_frame(item):
                    reason = "frozen"
                    break
        finally:
            self._kill(proc)
        if reason not in ("stop", "off_hours"):
            log.warning("[%s] sesi berakhir (%s) setelah %.0fs, %d frame. ffmpeg: %s",
                        self.cam_id, reason, time.time() - start, got, " | ".join(errs) or "-")
        self.session_ok = got > 0 and (time.time() - start) > 60
        return reason

    @staticmethod
    def _drain_err(proc, errs):
        try:
            for line in iter(proc.stderr.readline, b""):
                errs.append(line.decode("utf-8", "replace").strip()[:200])
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_jpegs(proc, q):
        """Pecah aliran MJPEG dari ffmpeg menjadi JPEG satuan (penanda FFD8 ... FFD9)."""
        buf = bytearray()
        try:
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(b"\xff\xd8")
                    if s < 0:
                        del buf[:-1]  # sisakan 1 byte kalau penanda terpotong di batas chunk
                        break
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e < 0:
                        del buf[:s]
                        break
                    q.put(bytes(buf[s:e + 2]))
                    del buf[:e + 2]
        except Exception:  # noqa: BLE001
            pass
        finally:
            q.put(None)

    @staticmethod
    def _kill(proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def run(self):
        log.info("[%s] mulai, mode=%s, jarak sampel=%.1fs -> %s", self.cam_id, self.mode,
                 self.sample_period, self.out_dir)
        backoff = self.backoff0
        self.session_ok = False
        while not self.stop_evt.is_set():
            if not in_hours(self._now(), self.hours):
                self.stop_evt.wait(30)
                continue
            reason = self._session()
            if reason in ("stop", "off_hours"):
                continue
            self.stats["restarts"] += 1
            backoff = self.backoff0 if self.session_ok else min(self.backoff_max, backoff * 2)
            self.stop_evt.wait(backoff + random.uniform(0, 2))


# --------------------------------------------------------------------------
# Worker klip video per kamera
# --------------------------------------------------------------------------
class ClipWorker(threading.Thread):
    def __init__(self, cam_id, cam, cfg, ffmpeg, stop, index):
        super().__init__(name=f"clip-{cam_id}", daemon=True)
        c = cfg["capture"]
        k = c["clips"]
        self.cam_id, self.url, self.ffmpeg, self.stop_evt = cam_id, cam["url"], ffmpeg, stop
        self.tz = ZoneInfo(c["timezone"])
        self.hours = parse_hours(c.get("active_hours"))
        self.dir = resolve_dir(c["clips_dir"]) / cam_id
        self.dur = int(k["duration_s"])
        self.every = int(k["every_min"]) * 60
        self.next_t = time.time() + 30 + index * int(k["stagger_min"]) * 60
        self.count = 0

    def run(self):
        while not self.stop_evt.wait(5):
            if time.time() < self.next_t:
                continue
            if not in_hours(datetime.now(self.tz), self.hours):
                self.next_t = time.time() + 60
                continue
            started = time.time()
            self._record()
            self.next_t = started + self.every

    def _record(self):
        now = datetime.now(self.tz)
        d = self.dir / now.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        final = d / (now.strftime("%H%M%S") + ".mp4")
        part = final.with_suffix(".mp4.part")
        cmd = [self.ffmpeg, *FFMPEG_INPUT, "-i", self.url, "-t", str(self.dur),
               "-map", "0:v:0", "-c", "copy", "-f", "mp4", "-y", str(part)]
        log.info("[%s] rekam klip %ds -> %s", self.cam_id, self.dur, final.name)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, **popen_flags())
        deadline = time.time() + self.dur + 90
        while proc.poll() is None:
            if self.stop_evt.is_set() or time.time() > deadline:
                try:  # 'q' = ffmpeg menutup file dengan rapi supaya klip tetap bisa diputar
                    proc.stdin.write(b"q")
                    proc.stdin.flush()
                    proc.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    proc.kill()
                break
            time.sleep(1)
        ok = part.exists() and part.stat().st_size > 100_000
        if ok:
            part.replace(final)
            self.count += 1
            log.info("[%s] klip selesai: %s (%.1f MB)", self.cam_id, final.name, final.stat().st_size / 1e6)
        else:
            err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()[-200:]
            log.warning("[%s] klip gagal/terlalu kecil. ffmpeg: %s", self.cam_id, err or "-")
            part.unlink(missing_ok=True)  # hanya file .part milik kita sendiri, bukan data mentah


# --------------------------------------------------------------------------
# Mode --snapshot: tes URL + simpan frame referensi (untuk membuat ROI)
# --------------------------------------------------------------------------
def snapshot(cfg, ffmpeg, cam_ids) -> int:
    c = cfg["capture"]
    tz = ZoneInfo(c["timezone"])
    ref = resolve_dir(c["reference_dir"])
    ref.mkdir(parents=True, exist_ok=True)
    bad = 0
    for cam_id in cam_ids:
        out = ref / f"{cam_id}_{datetime.now(tz).strftime('%Y%m%d_%H%M%S')}.jpg"
        cmd = [ffmpeg, *FFMPEG_INPUT, "-i", cfg["cameras"][cam_id]["url"],
               "-frames:v", "1", "-update", "1", "-q:v", "2", str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=90, **popen_flags())
            ok = r.returncode == 0 and out.exists() and out.stat().st_size > 1000
            msg = f"{out.stat().st_size / 1024:.0f} KB -> {out}" if ok else r.stderr.decode("utf-8", "replace")[-300:]
        except subprocess.TimeoutExpired:
            ok, msg = False, "timeout 90 detik"
        print(f"[{'OK' if ok else 'GAGAL'}] {cam_id}: {msg}")
        bad += 0 if ok else 1
        time.sleep(2)  # jeda antar kamera
    return bad


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def setup_logging(log_dir: Path, tag: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(log_dir / f"capture_{tag}.log", maxBytes=5_000_000,
                                              backupCount=3, encoding="utf-8")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def main():
    ap = argparse.ArgumentParser(description="Capture frame & klip dari CCTV JakCCTV (Tahap 1).")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--node", help="nama node di config.yaml (mis. dihar / gerry) -> kamera mengikuti daftar node")
    ap.add_argument("--cameras", nargs="+", help="override: ID kamera tertentu saja")
    ap.add_argument("--mode", choices=["interval", "motion"], help="override mode dari config")
    ap.add_argument("--no-clips", action="store_true", help="matikan perekaman klip")
    ap.add_argument("--snapshot", action="store_true", help="ambil 1 frame per kamera lalu selesai (tes URL / frame referensi ROI)")
    ap.add_argument("--max-seconds", type=int, help="berhenti otomatis setelah N detik (untuk uji)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    c = cfg["capture"]
    if args.cameras:
        cam_ids = args.cameras
    elif args.node:
        if args.node not in cfg.get("nodes", {}):
            sys.exit(f"Node '{args.node}' tidak ada di config.yaml (bagian nodes).")
        cam_ids = cfg["nodes"][args.node]
    else:
        cam_ids = list(cfg["cameras"])
    unknown = [x for x in cam_ids if x not in cfg["cameras"]]
    if unknown:
        sys.exit(f"Kamera tidak dikenal: {unknown}. Pilihan: {list(cfg['cameras'])}")

    ffmpeg = find_ffmpeg(cfg)
    configure_hls_opts(ffmpeg)
    if args.snapshot:
        sys.exit(1 if snapshot(cfg, ffmpeg, cam_ids) else 0)

    tag = args.node or "all"
    setup_logging(resolve_dir(c["log_dir"]), tag)
    lock = acquire_lock(f"tj-capture-{tag}")
    if lock is None:
        print(f"capture untuk '{tag}' sudah berjalan, instance ini keluar.")
        return

    out_dir = resolve_dir(c["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    log.info("=== capture mulai | node=%s | kamera=%s | ffmpeg=%s", tag, cam_ids, ffmpeg)
    workers = [CameraWorker(i, cfg["cameras"][i], cfg, ffmpeg, stop, args.mode) for i in cam_ids]
    clippers = []
    if c["clips"]["enabled"] and not args.no_clips:
        clippers = [ClipWorker(i, cfg["cameras"][i], cfg, ffmpeg, stop, n) for n, i in enumerate(cam_ids)]
    for n, w in enumerate(workers):
        w.start()
        time.sleep(3)  # jangan menyambung semua kamera di detik yang sama
    for w in clippers:
        w.start()

    t0 = last_stat = time.time()
    min_free = float(c["min_free_disk_gb"]) * 1e9
    while not stop.wait(5):
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            log.info("batas --max-seconds tercapai, berhenti.")
            stop.set()
        if shutil.disk_usage(out_dir).free < min_free:
            log.critical("Sisa disk < %s GB -> capture dihentikan. Pindahkan/backup data dulu.", c["min_free_disk_gb"])
            stop.set()
        if time.time() - last_stat > 300:
            last_stat = time.time()
            for w in workers:
                log.info("[%s] ringkasan: %s", w.cam_id, w.stats)
    log.info("menutup... (menunggu ffmpeg berhenti)")
    for w in workers + clippers:
        w.join(timeout=25)
    for w in workers:
        log.info("[%s] akhir: %s", w.cam_id, w.stats)
    log.info("=== capture selesai")


if __name__ == "__main__":
    main()
