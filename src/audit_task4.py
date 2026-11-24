"""Audit task 4_train_sisa: crop kotak truck & car besar (calon TJ yang salah kelas)
jadi grid bernomor. Dihar lihat grid, catat nomor yang sebenarnya bus.
Pakai: python src/audit_task4.py [--min_area_truck 0.0074] [--min_area_car 0.02]
Output: outputs/audit_task4/grid_XX.jpg + kandidat.csv (id, gambar, baris, kelas, area)."""
import argparse, csv, io, os, zipfile
from PIL import Image, ImageDraw

KELAS = {2: 'car', 4: 'truck'}
ap = argparse.ArgumentParser()
ap.add_argument('--min_area_truck', type=float, default=0.0074)  # p10 luas tj_bus hasil QC
ap.add_argument('--min_area_car', type=float, default=0.02)
ap.add_argument('--per_grid', type=int, default=48)
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), '..')
lab = zipfile.ZipFile(os.path.join(root, 'outputs/cvat/4_train_sisa_anotasi_yolo11.zip'))
img = zipfile.ZipFile(os.path.join(root, 'outputs/cvat/4_train_sisa_gambar.zip'))
imgs = {os.path.splitext(os.path.basename(n))[0]: n for n in img.namelist() if n.lower().endswith(('.jpg', '.png'))}
cand = []
for n in lab.namelist():
    if not (n.endswith('.txt') and '/' in n): continue
    stem = os.path.splitext(os.path.basename(n))[0]
    for i, l in enumerate(lab.read(n).decode().splitlines()):
        p = l.split()
        if len(p) != 5: continue
        c, x, y, w, h = int(p[0]), *map(float, p[1:])
        thr = {4: a.min_area_truck, 2: a.min_area_car}.get(c)
        if thr is not None and w * h >= thr:
            cand.append((stem, i, c, x, y, w, h))
cand.sort(key=lambda t: -t[5] * t[6])
out = os.path.join(root, 'outputs/audit_task4'); os.makedirs(out, exist_ok=True)
T, COLS = 200, 8
with open(os.path.join(out, 'kandidat.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['id', 'gambar', 'baris', 'kelas', 'area'])
    for k, (stem, i, c, *_r) in enumerate(cand):
        wr.writerow([k, stem, i, KELAS[c], round(_r[2] * _r[3], 4)])
for g in range(0, len(cand), a.per_grid):
    part = cand[g:g + a.per_grid]; rows = (len(part) + COLS - 1) // COLS
    sheet = Image.new('RGB', (COLS * T, rows * (T + 18)), 'white'); d = ImageDraw.Draw(sheet)
    for j, (stem, i, c, x, y, w, h) in enumerate(part):
        im = Image.open(io.BytesIO(img.read(imgs[stem]))).convert('RGB'); W, H = im.size
        pad = 0.15  # sedikit konteks di sekitar kotak
        box = (max(0, (x - w / 2 - w * pad) * W), max(0, (y - h / 2 - h * pad) * H),
               min(W, (x + w / 2 + w * pad) * W), min(H, (y + h / 2 + h * pad) * H))
        cr = im.crop(box); cr.thumbnail((T, T))
        cx, cy = (j % COLS) * T, (j // COLS) * (T + 18)
        sheet.paste(cr, (cx + (T - cr.width) // 2, cy + 18 + (T - cr.height) // 2))
        d.text((cx + 3, cy + 3), f'#{g + j} {KELAS[c]}', fill='red' if c == 4 else 'blue')
    sheet.save(os.path.join(out, f'grid_{g // a.per_grid:02d}.jpg'), quality=85)
print(len(cand), 'kandidat ->', out)
