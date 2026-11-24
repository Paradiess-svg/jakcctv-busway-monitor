"""figur_pemrosesan_citra.py -- gambar & angka Tahap 2 (pemrosesan citra) untuk laporan/presentasi.

Keluaran di outputs/laporan_citra/:
  01_pipeline_siang_malam.png   asli -> CLAHE -> CLAHE+gamma, siang & malam, + histogram kanal L
  02_zoom_malam.png             potongan diperbesar (detail kendaraan) asli vs hasil
  03_metode_lain.png            teknik lain di modul (bilateral, penekanan silau, unsharp) dan kenapa tidak dipakai
  04_kecerahan_siang_malam.png  sebaran kecerahan: kenapa deteksi siang/malam pakai jam, bukan ambang kecerahan
  05_hasil_model_c_vs_d.png     dampak ke deteksi YOLO: (c) tanpa vs (d) dengan CLAHE+gamma
  metrik_citra.csv              rata-rata kecerahan, kontras (std L), entropi sebelum/sesudah (data test)
  deteksi_siang_malam.csv       akurasi ambang kecerahan vs jam

Contoh: python src/figur_pemrosesan_citra.py
"""
import argparse, csv, os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import preprocessing as pp

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 110, "savefig.dpi": 160})


def baca_meta(ds):
    with open(os.path.join(ds, "meta.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def path_gambar(ds, r):
    for ext in (".jpg", ".png"):
        p = os.path.join(ds, "images", r["split"], r["file"] + ext)
        if os.path.exists(p):
            return p
    return None


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def kanal_L(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]


def entropi(g):
    h = np.bincount(g.ravel(), minlength=256).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def langkah(img, cfg, malam):
    """kembalikan dict tahap: asli, clahe, clahe+gamma (malam) -- sama persis dengan preprocessing.process()."""
    p = cfg["preprocessing"]
    c = pp.apply_clahe(img, p["clahe"]["clip_limit"], p["clahe"]["tile"])
    out = {"Asli": img, f"CLAHE (clip {p['clahe']['clip_limit']}, tile {p['clahe']['tile']}x{p['clahe']['tile']})": c}
    if malam:
        out[f"CLAHE + gamma {p['gamma_night']['value']} (malam)"] = pp.apply_gamma(c, p["gamma_night"]["value"])
    return out


def pilih(meta, ds, kamera, sm, split="test", jam_awal=None):
    kand = [r for r in meta if r["split"] == split and r["kamera"].startswith(kamera) and r["siang_malam"] == sm]
    if jam_awal:
        kand = [r for r in kand if r["jam"] >= jam_awal] or kand
    for r in kand:
        p = path_gambar(ds, r)
        if p:
            return r, cv2.imread(p)
    raise SystemExit(f"gambar {kamera} {sm} tidak ditemukan")


def fig_pipeline(contoh, cfg, out):
    fig, ax = plt.subplots(4, 3, figsize=(15, 13), gridspec_kw={"height_ratios": [3, 1.3, 3, 1.3]})
    for baris, (judul, r, img, malam) in enumerate(contoh):
        tahap = langkah(img, cfg, malam)
        a_img, a_his = ax[baris * 2], ax[baris * 2 + 1]
        warna = ["#64748b", "#2563eb", "#d97706"]
        for j in range(3):
            a_img[j].axis("off")
        for j, (nama, im) in enumerate(tahap.items()):
            a_img[j].imshow(rgb(im))
            L = kanal_L(im)
            a_img[j].set_title(f"{judul}: {nama}\nrata2 L {L.mean():.0f} · kontras (std L) {L.std():.1f}", fontsize=10)
            a_his[0 if j == 0 else 1].hist(L.ravel(), bins=64, range=(0, 255), histtype="step", lw=1.8, color=warna[j], label=nama.split(" (")[0])
        if not malam:
            a_img[2].text(.5, .5, "Siang: gamma TIDAK diterapkan\n(gamma hanya untuk malam,\nditentukan dari jam)", ha="center", va="center", fontsize=11, transform=a_img[2].transAxes, color="#475569")
        asli = kanal_L(img)
        a_his[0].hist(asli.ravel(), bins=64, range=(0, 255), color="#cbd5e1")
        a_his[0].set_title("Histogram kanal L asli", fontsize=10)
        a_his[1].set_title("Histogram kanal L: asli vs hasil (tumpuk)", fontsize=10)
        a_his[1].hist(asli.ravel(), bins=64, range=(0, 255), histtype="step", lw=1.8, color="#64748b", label="Asli")
        a_his[1].legend(fontsize=8)
        cdf = np.cumsum(np.bincount(asli.ravel(), minlength=256)); cdf2 = np.cumsum(np.bincount(kanal_L(list(tahap.values())[-1]).ravel(), minlength=256))
        a_his[2].plot(cdf / cdf[-1], color="#64748b", label="Asli"); a_his[2].plot(cdf2 / cdf2[-1], color=warna[len(tahap) - 1], label="Hasil akhir")
        a_his[2].set_title("CDF kecerahan (makin lurus = makin merata)", fontsize=10); a_his[2].legend(fontsize=8)
        for a in a_his:
            a.set_xlim(0, 255); a.spines[["top", "right"]].set_visible(False); a.set_yticks([]) if a is not a_his[2] else None
    fig.suptitle("Tahap 2 -- pemrosesan citra pada frame CCTV JakCCTV (siang & malam)", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_zoom(img, cfg, out, kotak):
    x0, y0, x1, y1 = kotak
    tahap = langkah(img, cfg, True)
    fig, ax = plt.subplots(1, len(tahap), figsize=(15, 4.6))
    for a, (nama, im) in zip(ax, tahap.items()):
        a.imshow(rgb(im[y0:y1, x0:x1]), interpolation="nearest"); a.set_title(nama); a.axis("off")
    fig.suptitle("Potongan diperbesar (malam): CLAHE menaikkan kontras lokal, gamma mencerahkan area gelap tapi ikut memperkuat noise & silau", fontsize=11)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_metode_lain(img, cfg, out, kotak):
    x0, y0, x1, y1 = kotak
    c = img[y0:y1, x0:x1]
    calon = [("Asli", c, "acuan"),
             ("Denoise bilateral (d=9, σ=75)", pp.apply_denoise(c, "bilateral"), "MATI: noise turun, tapi motor/mobil jauh ikut kabur"),
             ("Penekanan silau (V > 240 → ×0,8)", pp.suppress_glare(c, 240), "MATI: lampu kendaraan sedikit redup, efek ke deteksi kecil"),
             ("Unsharp masking (amount 0,5)", pp.apply_unsharp(c, 0.5), "MATI: tepi tajam, tapi noise kompresi CCTV ikut tajam")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))
    for a, (n, im, k) in zip(ax, calon):
        a.imshow(rgb(im), interpolation="nearest"); a.set_title(n, fontsize=10); a.set_xlabel(k, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Teknik lain yang ada di modul preprocessing.py (bisa dinyalakan lewat config, tapi DIMATIKAN)", fontsize=12)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def kecerahan_semua(meta, ds):
    hasil = []
    for r in meta:
        p = path_gambar(ds, r)
        if not p:
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        hasil.append((r, float(g.mean())))
    return hasil


def fig_kecerahan(data, thr, out, out_csv):
    s = [b for r, b in data if r["siang_malam"] == "siang"]
    m = [b for r, b in data if r["siang_malam"] == "malam"]
    benar_thr = sum(b >= thr for b in s) + sum(b < thr for b in m)
    m_kena = sum(b < thr for b in m)
    # ambang terbaik (kalau tetap mau pakai kecerahan)
    terbaik = max(range(0, 256), key=lambda t: sum(b >= t for b in s) + sum(b < t for b in m))
    akur_terbaik = (sum(b >= terbaik for b in s) + sum(b < terbaik for b in m)) / len(data)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metode", "ambang", "gambar", "siang", "malam", "malam_terdeteksi_malam", "akurasi"])
        w.writerow(["ambang kecerahan (Ivan)", thr, len(data), len(s), len(m), m_kena, round(benar_thr / len(data), 3)])
        w.writerow(["ambang kecerahan terbaik", terbaik, len(data), len(s), len(m), sum(b < terbaik for b in m), round(akur_terbaik, 3)])
        w.writerow(["jam (>=18:00 atau <05:30)", "-", len(data), len(s), len(m), len(m), 1.0])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.hist(s, bins=50, range=(0, 200), alpha=.65, color="#f59e0b", label=f"Siang (n={len(s)}), median {np.median(s):.0f}")
    ax.hist(m, bins=50, range=(0, 200), alpha=.65, color="#1e3a8a", label=f"Malam (n={len(m)}), median {np.median(m):.0f}")
    ax.axvline(thr, color="#dc2626", ls="--", lw=2)
    ax.text(thr + 2, ax.get_ylim()[1] * .9, f"ambang {thr}: hanya {m_kena}/{len(m)}\ngambar malam terdeteksi malam", color="#dc2626", fontsize=9, va="top")
    ax.set_xlabel("Rata-rata kecerahan gambar (abu-abu 0-255)"); ax.set_ylabel("Jumlah gambar")
    ax.set_title("Kenapa siang/malam ditentukan dari JAM: malam CCTV Jakarta terang (lampu jalan + auto-exposure)")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return dict(n=len(data), s=len(s), m=len(m), m_kena=m_kena, akur_thr=benar_thr / len(data), terbaik=terbaik, akur_terbaik=akur_terbaik,
                med_s=float(np.median(s)), med_m=float(np.median(m)))


def metrik_citra(meta, ds, cfg, out_csv):
    baris = []
    for r in meta:
        if r["split"] != "test":
            continue
        p = path_gambar(ds, r)
        if not p:
            continue
        img = cv2.imread(p); malam = r["siang_malam"] == "malam"
        hasil, _ = pp.process(img, cfg, r["jam"])
        for nama, im in (("asli", img), ("hasil", hasil)):
            L = kanal_L(im)
            baris.append((r["siang_malam"], nama, L.mean(), L.std(), entropi(L)))
    ring = []
    for sm in ("siang", "malam"):
        for nama in ("asli", "hasil"):
            d = np.array([b[2:] for b in baris if b[0] == sm and b[1] == nama])
            ring.append([sm, nama, len(d)] + [round(float(x), 2) for x in d.mean(0)])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["subset", "versi", "gambar", "rata2_kecerahan_L", "kontras_std_L", "entropi_bit"]); w.writerows(ring)
    return ring


def fig_hasil(out):
    # angka dari hasil_training_v2 / hasil_training_CLAHE (metrik_per_kelas.csv) dan outputs/evaluasi_v2{,_clahe}/tj_vs_bukan.csv
    sub = ["Valid", "Test", "Test siang", "Test malam", "Valid malam"]
    c = [0.737, 0.750, 0.800, 0.675, 0.642]; d = [0.737, 0.745, 0.796, 0.679, 0.615]
    uk = ["Recall TJ\n(test)", "Presisi TJ\n(test)", "Recall non-TJ\n(test)", "Recall non-TJ\n(test malam)"]
    c2 = [0.899, 0.872, 0.807, 0.738]; d2 = [0.899, 0.879, 0.763, 0.678]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a, lab, x1, x2, judul in ((ax[0], sub, c, d, "mAP50 rata-rata 5 kelas utama"), (ax[1], uk, c2, d2, "TJ vs bukan-TJ (conf 0,25, IoU 0,5)")):
        i = np.arange(len(lab)); w = .38
        b1 = a.bar(i - w / 2, x1, w, color="#64748b", label="(c) tanpa preprocessing")
        b2 = a.bar(i + w / 2, x2, w, color="#2563eb", label="(d) CLAHE + gamma malam")
        for bb in (b1, b2):
            a.bar_label(bb, fmt="%.3f", fontsize=8, padding=2)
        a.set_xticks(i, lab); a.set_ylim(0.5, 1.0); a.set_title(judul); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Dampak preprocessing ke deteksi YOLO11s: tidak ada peningkatan berarti (selisih < 0,01, arah tidak konsisten)", fontsize=12)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(AKAR, "data", "dataset", "v2"))
    ap.add_argument("--out", default=os.path.join(AKAR, "outputs", "laporan_citra"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = pp.load_config(os.path.join(AKAR, "config", "config.yaml"))
    meta = baca_meta(a.dataset)

    rs, siang = pilih(meta, a.dataset, "c04", "siang")
    rm, malam = pilih(meta, a.dataset, "c01", "malam", jam_awal="18:51")
    print("contoh siang:", rs["file"], "| contoh malam:", rm["file"])
    fig_pipeline([("Siang", rs, siang, False), ("Malam", rm, malam, True)], cfg, os.path.join(a.out, "01_pipeline_siang_malam.png"))
    h, w = malam.shape[:2]
    kotak = (int(w * .45), int(h * .35), int(w * .95), int(h * .95))
    fig_zoom(malam, cfg, os.path.join(a.out, "02_zoom_malam.png"), kotak)
    fig_metode_lain(malam, cfg, os.path.join(a.out, "03_metode_lain.png"), kotak)
    k = fig_kecerahan(kecerahan_semua(meta, a.dataset), cfg["preprocessing"]["day_night_threshold"],
                      os.path.join(a.out, "04_kecerahan_siang_malam.png"), os.path.join(a.out, "deteksi_siang_malam.csv"))
    print("kecerahan:", k)
    print("metrik test:", metrik_citra(meta, a.dataset, cfg, os.path.join(a.out, "metrik_citra.csv")))
    fig_hasil(os.path.join(a.out, "05_hasil_model_c_vs_d.png"))
    print("selesai ->", a.out)


if __name__ == "__main__":
    main()
