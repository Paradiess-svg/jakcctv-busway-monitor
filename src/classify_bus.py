"""classify_bus.py -- Tugas 3.6: pilah kotak bus hasil autolabel jadi tj_bus / bus_lain / dll.

Cara kerja (semi-otomatis, manusia memutuskan per CLUSTER, bukan per kotak):
  1) embed   : crop semua kotak kelas 1 (bus_lain) -> embedding CLIP (512 angka "rupa" per crop)
  2) klaster : K-Means mengelompokkan crop yang mirip -> grid contoh per cluster (gambar)
               --pecah X : cluster X yang isinya campur dipecah lagi jadi Xa, Xb, ...
  3) terapkan: Dihar memberi label tiap cluster -> salin labels/ ke labels_v2/ dengan kelas baru.
               Label asli TIDAK ditimpa.

Label per cluster yang boleh dipakai di --peta:
  tj -> 0 tj_bus | lain -> 1 bus_lain | truck -> 4 truck | mikrotrans -> 5 | prioritas -> 6 kend_prioritas
  ragu -> tetap 1, ditandai "cek_qc" di perubahan_3_6.csv
Crop terlalu kecil (sisi pendek < --min-px) tidak di-embed: tetap kelas 1, ditandai "kecil" (cek saat QC).

Hasil ini PRE-LABEL; tetap wajib QC manusia (3.7). Jangan klaim sebagai ground truth.

Contoh (Kaggle, GPU + Internet On):
  python classify_bus.py embed    --images IMG --labels LBL --out /kaggle/working/k36
  python classify_bus.py klaster  --out /kaggle/working/k36 --k 8
  python classify_bus.py klaster  --out /kaggle/working/k36 --pecah 4 --k 3     # opsional
  python classify_bus.py terapkan --labels LBL --out /kaggle/working/k36 --peta "0=tj,1=lain,2=tj,3=truck,..."
IMG = folder berisi images/<split>/, LBL = folder berisi labels/<split>/ (hasil autolabel.py)
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

NAMA_KELAS = ["tj_bus", "bus_lain", "car", "motorcycle", "truck", "mikrotrans", "kend_prioritas"]
PETA_LABEL = {"tj": 0, "lain": 1, "truck": 4, "mikrotrans": 5, "prioritas": 6, "ragu": 1}
KELAS_BUS = 1
EKSTENSI = (".jpg", ".jpeg", ".png")


def jam_dari_nama(nama: str) -> int:
    """c01_taman_literasi__2026-10-29__193512.jpg -> 19"""
    try:
        return int(Path(nama).stem.split("__")[-1][:2])
    except ValueError:
        return -1


def cari_gambar(img_root: Path, split: str, stem: str):
    for e in EKSTENSI:
        p = img_root / split / (stem + e)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- 1) embed
def cmd_embed(a):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    img_root, lbl_root, out = Path(a.images), Path(a.labels), Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    baris, kecil = [], 0
    for lab in sorted(lbl_root.rglob("*.txt")):
        split = lab.parent.name
        isi = lab.read_text(encoding="utf-8").strip().splitlines()
        bus = [(k, l) for k, l in enumerate(isi) if l.split()[0] == str(KELAS_BUS)]
        if not bus:
            continue
        gp = cari_gambar(img_root, split, lab.stem)
        if gp is None:
            print(f"  [peringatan] gambar untuk {split}/{lab.stem} tidak ketemu, dilewati")
            continue
        im = Image.open(gp).convert("RGB")
        W, H = im.size
        for k, l in bus:
            _, x, y, w, h = map(float, l.split())
            bw, bh = w * W, h * H
            if min(bw, bh) < a.min_px:
                kecil += 1
                continue
            pad = a.pad
            x1, y1 = max(0, (x - w / 2 - w * pad) * W), max(0, (y - h / 2 - h * pad) * H)
            x2, y2 = min(W, (x + w / 2 + w * pad) * W), min(H, (y + h / 2 + h * pad) * H)
            cid = f"{split}__{lab.stem}__b{k}"
            im.crop((int(x1), int(y1), int(x2), int(y2))).save(out / "crops" / f"{cid}.jpg", quality=92)
            baris.append({"crop_id": cid, "split": split, "gambar": lab.stem, "baris": k,
                          "lebar_px": round(bw), "tinggi_px": round(bh), "jam": jam_dari_nama(lab.stem)})

    if not baris:
        raise SystemExit("Tidak ada kotak bus yang bisa di-crop. Cek --labels dan --images.")
    print(f"{len(baris)} crop bus dibuat, {kecil} kotak terlalu kecil (< {a.min_px}px) dilewati")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Memuat CLIP {a.model} di {dev} ...")
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model)

    emb = []
    with torch.no_grad():
        for i in range(0, len(baris), a.batch):
            ims = [Image.open(out / "crops" / f"{r['crop_id']}.jpg").convert("RGB") for r in baris[i:i + a.batch]]
            inp = proc(images=ims, return_tensors="pt").to(dev)
            f = model.get_image_features(**inp)
            if not torch.is_tensor(f):  # beberapa versi transformers mengembalikan objek, bukan tensor
                f = f.image_embeds if getattr(f, "image_embeds", None) is not None else f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            emb.append(f.cpu().numpy())
            print(f"  embed {min(i + a.batch, len(baris))}/{len(baris)}")
    np.save(out / "emb.npy", np.concatenate(emb).astype("float32"))

    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(baris[0]))
        w.writeheader()
        w.writerows(baris)
    (out / "kecil.txt").write_text(str(kecil), encoding="utf-8")
    print(f"Selesai: {out/'emb.npy'}, {out/'crops.csv'}")


# ---------------------------------------------------------------- 2) klaster
def baca_crops(out: Path):
    with open(out / "crops.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def baca_klaster(out: Path):
    p = out / "klaster.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return {r["crop_id"]: r["cluster"] for r in csv.DictReader(fh)}


def buat_grid(out: Path, nama_cluster: str, anggota: list, n=25, ukuran=128, seed=0):
    from PIL import Image, ImageDraw

    contoh = list(anggota)
    random.Random(seed).shuffle(contoh)
    contoh = contoh[:n]
    kol = 5
    brs = (len(contoh) + kol - 1) // kol
    kepala = 44
    kanvas = Image.new("RGB", (kol * ukuran, kepala + brs * ukuran), (30, 30, 30))
    malam = sum(1 for r in anggota if int(r["jam"]) >= 18)
    per_split = Counter(r["split"] for r in anggota)
    d = ImageDraw.Draw(kanvas)
    d.text((6, 4), f"CLUSTER {nama_cluster}  |  {len(anggota)} crop  |  malam {malam}", fill=(255, 255, 0))
    d.text((6, 22), "  ".join(f"{s}:{c}" for s, c in sorted(per_split.items())), fill=(200, 200, 200))
    for i, r in enumerate(contoh):
        t = Image.open(out / "crops" / f"{r['crop_id']}.jpg").convert("RGB")
        t.thumbnail((ukuran - 4, ukuran - 4))
        x = (i % kol) * ukuran + (ukuran - t.width) // 2
        y = kepala + (i // kol) * ukuran + (ukuran - t.height) // 2
        kanvas.paste(t, (x, y))
    (out / "grid").mkdir(exist_ok=True)
    p = out / "grid" / f"cluster_{nama_cluster}.jpg"
    kanvas.save(p, quality=90)
    return p


def cmd_klaster(a):
    import numpy as np
    from sklearn.cluster import KMeans

    out = Path(a.out)
    crops = baca_crops(out)
    emb = np.load(out / "emb.npy")
    lama = baca_klaster(out)

    if a.pecah is None:
        lab = KMeans(n_clusters=a.k, n_init=10, random_state=0).fit_predict(emb)
        tugas = {r["crop_id"]: str(int(c)) for r, c in zip(crops, lab)}
        if (out / "grid").exists():
            shutil.rmtree(out / "grid")
    else:
        if lama is None:
            raise SystemExit("Jalankan 'klaster' tanpa --pecah dulu.")
        idx = [i for i, r in enumerate(crops) if lama[r["crop_id"]] == a.pecah]
        if len(idx) < a.k:
            raise SystemExit(f"Cluster {a.pecah} cuma {len(idx)} anggota, terlalu sedikit untuk dipecah jadi {a.k}.")
        sub = KMeans(n_clusters=a.k, n_init=10, random_state=0).fit_predict(emb[idx])
        tugas = dict(lama)
        for i, c in zip(idx, sub):
            tugas[crops[i]["crop_id"]] = f"{a.pecah}{'abcdefghij'[c]}"
        g = out / "grid" / f"cluster_{a.pecah}.jpg"
        if g.exists():
            g.unlink()

    with open(out / "klaster.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["crop_id", "cluster"])
        w.writerows((r["crop_id"], tugas[r["crop_id"]]) for r in crops)

    grup = defaultdict(list)
    for r in crops:
        grup[tugas[r["crop_id"]]].append(r)
    baru = sorted(grup) if a.pecah is None else sorted(c for c in grup if c.startswith(a.pecah) and c != a.pecah)
    for c in baru:
        buat_grid(out, c, grup[c])
    print("Cluster sekarang (nama: jumlah crop):")
    for c in sorted(grup, key=lambda s: (len(s), s)):
        print(f"  {c}: {len(grup[c])}")
    print(f"Grid ada di {out/'grid'}")


# ---------------------------------------------------------------- 3) terapkan
def parse_peta(teks: str):
    peta = {}
    for bagian in teks.replace(" ", "").split(","):
        if not bagian:
            continue
        c, lbl = bagian.split("=")
        if lbl not in PETA_LABEL:
            raise SystemExit(f"Label '{lbl}' untuk cluster {c} tidak dikenal. Pilihan: {', '.join(PETA_LABEL)}")
        peta[c] = lbl
    return peta


def cmd_terapkan(a):
    lbl_root, out = Path(a.labels), Path(a.out)
    crops = {r["crop_id"]: r for r in baca_crops(out)}
    klaster = baca_klaster(out)
    if klaster is None:
        raise SystemExit("klaster.csv belum ada. Jalankan 'klaster' dulu.")
    peta = parse_peta(a.peta)
    belum = sorted(set(klaster.values()) - set(peta))
    if belum:
        raise SystemExit(f"Cluster ini belum diberi label di --peta: {', '.join(belum)}")

    # (split, gambar, baris) -> (cluster, label)
    keputusan = {}
    for cid, c in klaster.items():
        r = crops[cid]
        keputusan[(r["split"], r["gambar"], int(r["baris"]))] = (c, peta[c])

    tujuan = out / "labels_v2"
    if tujuan.exists():
        shutil.rmtree(tujuan)
    log, hitung = [], defaultdict(Counter)
    for lab in sorted(lbl_root.rglob("*.txt")):
        split = lab.parent.name
        isi = lab.read_text(encoding="utf-8").strip().splitlines()
        hasil = []
        for k, l in enumerate(isi):
            t = l.split()
            if t[0] == str(KELAS_BUS):
                c, lbl = keputusan.get((split, lab.stem, k), ("-", None))
                if lbl is None:
                    log.append([split, lab.stem, k, "-", "kecil", NAMA_KELAS[KELAS_BUS], "cek_qc"])
                else:
                    t[0] = str(PETA_LABEL[lbl])
                    log.append([split, lab.stem, k, c, lbl, NAMA_KELAS[int(t[0])], "cek_qc" if lbl == "ragu" else ""])
            hitung[split][NAMA_KELAS[int(t[0])]] += 1
            hasil.append(" ".join(t))
        p = tujuan / split / lab.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(hasil) + ("\n" if hasil else ""), encoding="utf-8", newline="\n")

    (out / "classes.txt").write_text("\n".join(NAMA_KELAS) + "\n", encoding="utf-8")
    with open(out / "perubahan_3_6.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "gambar", "baris", "cluster", "label_cluster", "kelas_baru", "catatan"])
        w.writerows(log)
    teks = [f"[{s}] {dict(sorted(hitung[s].items()))}" for s in sorted(hitung)]
    teks.append(f"Ditandai cek_qc (ragu + kecil): {sum(1 for r in log if r[-1] == 'cek_qc')}")
    (out / "ringkasan_v2.txt").write_text("\n".join(teks) + "\n", encoding="utf-8")
    print("\n".join(teks))
    print(f"Selesai: {tujuan}, {out/'perubahan_3_6.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    e = sp.add_parser("embed", help="crop bus + hitung embedding CLIP")
    e.add_argument("--images", required=True, help="folder berisi <split>/ gambar")
    e.add_argument("--labels", required=True, help="folder berisi <split>/ label hasil autolabel")
    e.add_argument("--out", required=True)
    e.add_argument("--model", default="openai/clip-vit-base-patch16")
    e.add_argument("--min-px", type=int, default=20, help="sisi pendek kotak minimal (px) supaya di-embed")
    e.add_argument("--pad", type=float, default=0.05, help="tambahan tepi crop (fraksi ukuran kotak)")
    e.add_argument("--batch", type=int, default=64)
    e.set_defaults(fn=cmd_embed)

    k = sp.add_parser("klaster", help="K-Means + grid per cluster")
    k.add_argument("--out", required=True)
    k.add_argument("--k", type=int, default=8, help="jumlah cluster (atau jumlah pecahan kalau --pecah)")
    k.add_argument("--pecah", default=None, help="nama cluster yang mau dipecah lagi, mis. 4")
    k.set_defaults(fn=cmd_klaster)

    t = sp.add_parser("terapkan", help="tulis labels_v2/ sesuai label per cluster")
    t.add_argument("--labels", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--peta", required=True, help='mis. "0=tj,1=lain,2=tj,3=truck,4a=tj,4b=ragu"')
    t.set_defaults(fn=cmd_terapkan)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
