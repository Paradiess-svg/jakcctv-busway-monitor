"""server.py -- Tahap 6: backend dasbor monitoring TJ vs non-TJ (FastAPI). Bukan sistem tilang.

Jalankan (dari folder Project-TJ):
  python dashboard/server.py            -> http://localhost:8000      (halaman uji sederhana)
                                          http://localhost:8000/docs (dokumentasi API otomatis)
Kontrak API untuk tim desain/frontend: dashboard/API.md

Dua mode:
  live   : stream HLS JakCCTV (maks 4 kamera, grid) -> dashboard/live.py
  upload : unggah video + pilih kamera (untuk ROI) -> diproses di latar belakang dengan src/monitor.py olah_video()
"""
import csv
import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if hasattr(time, "tzset"):              # Linux (Kaggle/Colab jamnya UTC) -> pakai WIB untuk log & siang/malam
    os.environ["TZ"] = os.environ.get("TJ_TZ", "Asia/Jakarta")
    time.tzset()
TOKEN = os.environ.get("TJ_TOKEN", "")  # kalau diisi (mode online): aksi ubah (POST) butuh header X-Token
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live import LiveManager  # noqa: E402
from monitor import kamera_dari_path, muat_config, olah_video  # noqa: E402

CFG = muat_config()
OUT = ROOT / CFG["output_dir"]
OUT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = OUT / "upload"
MAKS_UPLOAD_MB = 500

app = FastAPI(title="Monitoring Busway TJ vs non-TJ", version="1.0",
              description="Deteksi & klasifikasi kendaraan di jalur busway dari CCTV publik. Tanpa penilangan.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=str(OUT)), name="files")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
live = LiveManager()
DASBOR = live.dasbor
VERIF = OUT / "verifikasi.json"   # hasil cek manusia per kejadian: {id_snapshot: "benar" | "salah"}
_kunci_verif = threading.Lock()


@app.middleware("http")
async def tandai_dibuka(request, call_next):
    if request.url.path.startswith("/api/"):
        live.sentuh()   # dasbor masih dibuka -> live jangan dihentikan
        if TOKEN and request.method == "POST" and request.headers.get("x-token") != TOKEN:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Mode lihat saja. Buka link admin untuk mengubah."}, status_code=403)
    return await call_next(request)


@app.get("/api/info", summary="Mode server: gpu, butuh token admin")
def info():
    return {"gpu": live.gpu, "butuh_token": bool(TOKEN), "mode": _cfg_dasbor().get("mode", {"semua": []})}


@app.get("/thumb/{path:path}", include_in_schema=False)
def thumb(path: str):
    """Snapshot versi kecil (240 px) untuk tabel log, disimpan di outputs/monitor/_thumb (web jauh lebih ringan)."""
    src = (OUT / path).resolve()
    if OUT.resolve() not in src.parents or not src.is_file():
        raise HTTPException(404, "tidak ada")
    dst = OUT / "_thumb" / path
    if not dst.exists():
        im = cv2.imread(str(src))
        if im is None:
            raise HTTPException(404, "gambar rusak")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.resize(im, (240, 135), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return FileResponse(dst, headers={"Cache-Control": "max-age=86400"})


def _tampilan(k):
    return DASBOR.get("tampilan", {}).get(k, {})


# ======================= umum =======================
@app.get("/api/kamera", summary="Daftar kamera: nama, lokasi (perkiraan), aktif, ROI ada")
def daftar_kamera(semua: bool = False):
    hasil = []
    for k, v in live.kamera_cfg.items():
        aktif = v.get("aktif", True)
        if not aktif and not semua:
            continue
        t = _tampilan(k)
        hasil.append({"id": k, "nama": t.get("nama", k), "nama_lengkap": v.get("name", k), "lokasi": t.get("lokasi"),
                      "aktif": aktif, "roi": (ROOT / CFG["roi_dir"] / f"{k}.json").exists(),
                      "default": k in DASBOR.get("kamera_default", [])})
    return hasil


@app.get("/api/config", summary="Aturan pencatatan yang sedang dipakai")
def config():
    return {k: CFG[k] for k in ("model", "conf", "proses_fps", "min_detik_di_roi", "kelompok", "preprocessing")}


@app.get("/api/kamera/{kamera}/roi", summary="Poligon ROI (koordinat frame 640x360)")
def roi(kamera: str):
    import json
    p = ROOT / CFG["roi_dir"] / f"{kamera}.json"
    if not p.exists():
        raise HTTPException(404, "ROI tidak ada")
    return json.load(open(p, encoding="utf-8"))


# ======================= live =======================
class LiveStart(BaseModel):
    kamera: list[str] = []


@app.post("/api/live/start", summary="Mulai live (1-4 kamera). Sesi lama dihentikan dulu.")
def live_start(body: LiveStart):
    if not body.kamera:
        body.kamera = DASBOR.get("kamera_default", [])
    if not 1 <= len(body.kamera) <= 4:
        raise HTTPException(400, "pilih 1-4 kamera")
    try:
        live.start(body.kamera)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return live.status()


@app.get("/api/live/bangun", summary="Dasbor dibuka & live sedang mati semua -> nyalakan lagi grid terakhir/default")
def live_bangun():
    if not live.pembaca:
        daftar = live.daftar or DASBOR.get("kamera_default", [])
        try:
            live.start(daftar)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return live.status()


@app.post("/api/live/{kamera}/nyala", summary="Nyalakan / sambung ulang (reset) satu kamera")
def live_nyala(kamera: str):
    try:
        live.nyalakan(kamera)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return live.status()


@app.post("/api/live/{kamera}/mati", summary="Matikan satu kamera (pencatatan kamera ini berhenti)")
def live_mati(kamera: str):
    live.matikan(kamera)
    return live.status()


@app.post("/api/live/stop", summary="Hentikan live")
def live_stop():
    live.stop()
    return {"aktif": False}


@app.get("/api/live/status", summary="Status tiap kamera: stream, fps analisis, hitungan TJ/non-TJ")
def live_status():
    return live.status()


@app.get("/api/live/{kamera}/frame.jpg", summary="Frame beranotasi terbaru (JPEG)")
def live_frame(kamera: str):
    jpg = live.jpeg_terbaru(kamera)
    if jpg is None:
        raise HTTPException(404, "belum ada frame")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/live/{kamera}/mjpeg", summary="Video beranotasi (MJPEG) -> cukup <img src=...>")
def live_mjpeg(kamera: str):
    def gen():
        lalu = None
        while kamera in live.pembaca:
            live.sentuh()
            jpg = live.jpeg_terbaru(kamera)
            if jpg is not None and jpg is not lalu:
                lalu = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + \
                      b"\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.1)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/live/log", summary="Kejadian live (polling: kirim setelah_id = id terakhir yang sudah diterima)")
def live_log(setelah_id: int = 0, kelompok: str | None = Query(None, pattern="^(tj|non_tj)$"),
             kamera: str | None = None, batas: int = 200):
    return live.kejadian_sejak(setelah_id, kelompok, kamera, batas)


# ======================= upload =======================
JOB: dict[str, dict] = {}


def _jalankan_job(jid, video, kamera):
    j = JOB[jid]
    t0 = time.time()

    def progres(frac, tampil, mon):
        j["progres"] = round(frac, 3)
        j["hitung"] = {"tj": mon.hitung["tj"], "non_tj": mon.hitung["non_tj"]}
        j["kejadian"] = mon.kejadian
        if time.time() - j.get("_t_prev", 0) > 0.5:
            j["_t_prev"] = time.time()
            ok, jpg = cv2.imencode(".jpg", tampil, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                j["_preview"] = jpg.tobytes()

    try:
        mon, ringkas, out = olah_video(video, kamera, video.parent, video_out=True, progres=progres)
        j["progres"] = 0.999
        web = _ke_h264(out / "hasil.mp4")
        j.update(status="selesai", progres=1.0, ringkasan=ringkas, kejadian=mon.kejadian,
                 hitung={"tj": mon.hitung["tj"], "non_tj": mon.hitung["non_tj"]}, disaring_dobel=mon.dobel,
                 video_hasil=f"/files/upload/{jid}/{web.name}", waktu_proses_s=round(time.time() - t0))
    except BaseException as e:  # noqa: BLE001  (olah_video bisa SystemExit)
        j.update(status="gagal", error=str(e))


def _ke_h264(mp4: Path) -> Path:
    """VideoWriter OpenCV = mp4v, tidak bisa diputar browser -> ubah ke H.264 pakai ffmpeg (kalau ada)."""
    import subprocess
    web = mp4.with_name("hasil_web.mp4")
    try:
        subprocess.run([live.ffmpeg or shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(web)], check=True, timeout=900)
        mp4.unlink()
        return web
    except Exception:  # noqa: BLE001
        return mp4


@app.post("/api/upload", summary="Unggah video + ID kamera (menentukan ROI). Return job_id.")
async def upload(file: UploadFile = File(...), kamera: str | None = Form(None)):
    nama = Path(file.filename or "video.mp4").name
    if Path(nama).suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv", ".ts"):
        raise HTTPException(400, "format video tidak didukung")
    jid = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    d = UPLOAD_DIR / jid
    d.mkdir(parents=True)
    video = d / nama
    with open(video, "wb") as f:
        n = 0
        while chunk := await file.read(1 << 20):
            n += len(chunk)
            if n > MAKS_UPLOAD_MB << 20:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"maks {MAKS_UPLOAD_MB} MB")
            f.write(chunk)
    kamera = kamera or kamera_dari_path(Path(file.filename or ""))[0]
    if not kamera or not (ROOT / CFG["roi_dir"] / f"{kamera}.json").exists():
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "isi 'kamera' dengan ID kamera yang punya ROI (lihat /api/kamera)")
    cap = cv2.VideoCapture(str(video))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "video tidak bisa dibaca")
    JOB[jid] = {"job_id": jid, "kamera": kamera, "file": nama, "status": "proses", "progres": 0.0,
                "hitung": {"tj": 0, "non_tj": 0}, "kejadian": []}
    threading.Thread(target=_jalankan_job, args=(jid, video, kamera), daemon=True).start()
    return {"job_id": jid}


def _job(jid):
    if jid not in JOB:
        raise HTTPException(404, "job tidak ada")
    return JOB[jid]


def _url_kejadian(jid, ks):
    return [{**k, "snapshot_url": f"/files/upload/{jid}/{k['snapshot']}" if k.get("snapshot") else ""} for k in ks]


@app.get("/api/upload/{jid}", summary="Progres + hasil job upload")
def upload_status(jid: str):
    j = _job(jid)
    return {**{a: b for a, b in j.items() if not a.startswith("_") and a != "kejadian"},
            "kejadian": _url_kejadian(jid, j["kejadian"])}


@app.get("/api/upload/{jid}/preview.jpg", summary="Frame beranotasi terakhir selama diproses")
def upload_preview(jid: str):
    jpg = _job(jid).get("_preview")
    if jpg is None:
        raise HTTPException(404, "belum ada frame")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/upload", summary="Daftar job upload sesi server ini")
def upload_list():
    return [{a: b for a, b in j.items() if a in ("job_id", "kamera", "file", "status", "progres", "hitung")}
            for j in JOB.values()]


# ======================= statistik =======================
@app.get("/api/statistik", summary="Rekap kejadian (live sesi ini atau satu job upload): per kelas, kamera, jam")
def statistik(sumber: str = Query("live", description="'live' atau job_id upload")):
    ks = live.kejadian_sejak(0, batas=10 ** 9) if sumber == "live" else _job(sumber)["kejadian"]
    per_jam = Counter((k["waktu"][:13] + ":00", k["kelompok"]) for k in ks)
    return {"total": dict(Counter(k["kelompok"] for k in ks)),
            "per_kelas": dict(Counter(k["kelas"] for k in ks)),
            "per_kamera": {kam: dict(Counter(k["kelompok"] for k in ks if k["kamera"] == kam))
                           for kam in sorted({k["kamera"] for k in ks})},
            "per_jam": [{"jam": j, "kelompok": g, "jumlah": n} for (j, g), n in sorted(per_jam.items())]}


# ======================= riwayat log (semua sesi live, dari CSV) =======================
_cache_log = {"kunci": None, "data": []}


def _baca_semua_log():
    """Semua kejadian live dari outputs/monitor/live/<kamera>/<sesi>/log_*.csv, terbaru dulu.
    Sesi sebelum 19 Nov (tanpa kolom box) memakai jam dari waktu file snapshot, karena jam di CSV lama bisa
    tertinggal berjam-jam (waktu stream melambat saat stream putus-sambung)."""
    files = sorted((OUT / "live").glob("*/*/log_*.csv"))
    kunci = tuple((str(f), f.stat().st_mtime, f.stat().st_size) for f in files)
    if kunci == _cache_log["kunci"]:
        return _cache_log["data"]
    data = []
    for f in files:
        sesi = f.parent
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if not r.get("snapshot"):
                continue
            snap = sesi / r["snapshot"]
            rid = snap.relative_to(OUT).as_posix()
            waktu = r["waktu"]
            if not r.get("box") and snap.exists():
                waktu = datetime.fromtimestamp(snap.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            data.append({"id": rid, "waktu": waktu, "kamera": r["kamera"],
                         "nama_kamera": _tampilan(r["kamera"]).get("nama", r["kamera"]),
                         "kelas": r["kelas"], "kelompok": r["kelompok"], "conf_rata2": float(r["conf_rata2"]),
                         "snapshot_url": f"/files/{rid}"})
    data.sort(key=lambda e: e["waktu"], reverse=True)
    _cache_log.update(kunci=kunci, data=data)
    return data


def _verif():
    try:
        return json.load(open(VERIF, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _sembunyi(mode):
    """Kamera yang disembunyikan di mode studi kasus (config dasbor.mode); mode tak dikenal = semua tampil."""
    return set((_cfg_dasbor().get("mode") or {}).get(mode or "semua") or [])


def _filter(data, kelompok=None, kamera=None, tanggal=None, sembunyi=()):
    return [e for e in data if (not kelompok or e["kelompok"] == kelompok) and (not kamera or e["kamera"] == kamera)
            and (not tanggal or e["waktu"].startswith(tanggal)) and e["kamera"] not in sembunyi]


@app.get("/api/log", summary="Riwayat kejadian semua sesi live (terbaru dulu) + status verifikasi manusia")
def log_riwayat(kelompok: str | None = Query(None, pattern="^(tj|non_tj)$"), kamera: str | None = None,
                tanggal: str | None = Query(None, description="YYYY-MM-DD"),
                verifikasi: str | None = Query(None, pattern="^(benar|salah|belum)$"), batas: int = 100, offset: int = 0,
                mode: str = "semua"):
    v = _verif()
    d = [{**e, "verifikasi": v.get(e["id"])} for e in _filter(_baca_semua_log(), kelompok, kamera, tanggal, _sembunyi(mode))]
    if verifikasi:
        d = [e for e in d if (e["verifikasi"] or "belum") == verifikasi]
    return {"total": len(d), "data": d[offset:offset + batas]}


class Verifikasi(BaseModel):
    id: str
    status: str | None = None  # "benar", "salah", atau null (hapus)


@app.post("/api/verifikasi", summary="Tandai satu kejadian benar/salah (dicek manusia dari foto)")
def verifikasi(body: Verifikasi):
    if body.status not in ("benar", "salah", None):
        raise HTTPException(400, "status: benar / salah / null")
    with _kunci_verif:
        v = _verif()
        if body.status is None:
            v.pop(body.id, None)
        else:
            v[body.id] = body.status
        tmp = VERIF.with_suffix(".tmp")
        json.dump(v, open(tmp, "w", encoding="utf-8"), indent=0)
        tmp.replace(VERIF)
    return {"ok": True}


def _evaluasi_offline():
    """Angka uji model di gambar test berlabel (src/evaluasi.py, conf 0,25)."""
    p = ROOT / "outputs" / "evaluasi_v2" / "tj_vs_bukan.csv"
    try:
        rows = {r["subset"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
        return {s: {"recall_tj": float(rows[s]["recall_tj"]), "precision_tj": float(rows[s]["precision_tj"]),
                    "bukan_tj_jadi_tj_persen": round(100 * float(rows[s]["persen_bukan_tj_jadi_tj"]), 2)}
                for s in ("test", "test_siang", "test_malam") if s in rows}
    except Exception:  # noqa: BLE001
        return {}


@app.get("/api/akurasi", summary="Akurasi: (1) hasil verifikasi manusia di dasbor, (2) uji model di data test")
def akurasi(mode: str = "semua"):
    v = _verif()
    data = {e["id"]: e for e in _filter(_baca_semua_log(), sembunyi=_sembunyi(mode))}
    per = {}
    for rid, st in v.items():
        e = data.get(rid)
        if not e:
            continue
        for kunci in ("semua", e["kelompok"], e["kamera"]):
            d = per.setdefault(kunci, {"dicek": 0, "benar": 0})
            d["dicek"] += 1
            d["benar"] += st == "benar"
    for d in per.values():
        d["persen_benar"] = round(100 * d["benar"] / d["dicek"], 1) if d["dicek"] else None
    return {"verifikasi_manusia": per, "total_kejadian": len(data), "uji_model_data_test": _evaluasi_offline()}


def _cfg_dasbor():
    import yaml
    return yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8")).get("dasbor", {})


@app.get("/api/ringkasan", summary="Angka kartu dasbor + data grafik (periode: hari_ini / semua / YYYY-MM-DD)")
def ringkasan(periode: str = "hari_ini", kamera: str | None = None, mode: str = "semua"):
    tgl = datetime.now().strftime("%Y-%m-%d") if periode == "hari_ini" else None if periode == "semua" else periode
    sembunyi = _sembunyi(mode)
    d = _filter(_baca_semua_log(), kamera=kamera, tanggal=tgl, sembunyi=sembunyi)
    # Flyover Ladokgi bukan jalur busway (hanya pengganti CCTV yang rusak): tidak dihitung sebagai
    # pelanggaran / sterilisasi / total di jalur busway. Datanya tetap tampil di log, grafik, dan peta.
    kecuali = set(_cfg_dasbor().get("bukan_busway", [])) - sembunyi
    tot = Counter(e["kelompok"] for e in d if e["kamera"] not in kecuali)
    n = tot["tj"] + tot["non_tj"]
    st = live.status()
    aktif = [k for k, v in live.kamera_cfg.items() if v.get("aktif", True) and k not in sembunyi]
    per_jam = Counter((e["waktu"][11:13], e["kelompok"]) for e in d)
    return {"periode": periode, "tanggal": tgl, "total_kendaraan": n, "tj": tot["tj"], "non_tj": tot["non_tj"],
            "sterilisasi_persen": round(100 * tot["tj"] / n, 1) if n else None,
            "bukan_busway": kamera in kecuali if kamera else False, "kamera_dikecualikan": sorted(kecuali),
            "kamera_online": sum(k["stream"] == "hidup" for k in st["kamera"] if k["kamera"] not in sembunyi),
            "kamera_total": len(aktif), "mode": mode, "kamera": kamera,
            "per_kelas": dict(Counter(e["kelas"] for e in d).most_common()),
            "per_kamera": {k: dict(Counter(e["kelompok"] for e in d if e["kamera"] == k)) for k in aktif},
            "per_jam": [{"jam": f"{j:02d}", "tj": per_jam[(f"{j:02d}", "tj")], "non_tj": per_jam[(f"{j:02d}", "non_tj")]}
                        for j in range(24)]}


# ======================= halaman =======================
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"Dasbor: http://localhost:{a.port}  (Ctrl+C untuk berhenti; live berhenti sendiri kalau dasbor "
          f"tidak dibuka {DASBOR.get('live_idle_menit', 10)} menit)", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
