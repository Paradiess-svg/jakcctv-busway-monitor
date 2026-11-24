"""Lembar kontak gambar utuh task 4_train_sisa (737 gambar, 24/lembar, bernomor).
Tujuan: Dihar scan cepat, catat nomor gambar yang ADA bus TJ (salah kelas atau tak terdeteksi).
Gambar yang dicatat dibuang dari versi dataset 'dengan task 4'.
Pakai: python src/lembar_task4.py  -> outputs/audit_task4/lembar/lembar_XX.jpg + daftar_gambar.csv"""
import csv, io, os, zipfile
from PIL import Image, ImageDraw
root = os.path.join(os.path.dirname(__file__), '..')
z = zipfile.ZipFile(os.path.join(root, 'outputs/cvat/4_train_sisa_gambar.zip'))
names = sorted(n for n in z.namelist() if n.lower().endswith(('.jpg', '.png')))
out = os.path.join(root, 'outputs/audit_task4/lembar'); os.makedirs(out, exist_ok=True)
TW, TH, COLS, PER = 400, 225, 4, 24
with open(os.path.join(out, 'daftar_gambar.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['no', 'gambar'])
    for i, n in enumerate(names): w.writerow([i, os.path.basename(n)])
for g in range(0, len(names), PER):
    part = names[g:g + PER]; rows = (len(part) + COLS - 1) // COLS
    sheet = Image.new('RGB', (COLS * TW, rows * (TH + 4)), 'white'); d = ImageDraw.Draw(sheet)
    for j, n in enumerate(part):
        im = Image.open(io.BytesIO(z.read(n))).convert('RGB').resize((TW, TH))
        x, y = (j % COLS) * TW, (j // COLS) * (TH + 4)
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + 52, y + 16], fill='yellow'); d.text((x + 3, y + 2), f'#{g + j}', fill='black')
    sheet.save(os.path.join(out, f'lembar_{g // PER:02d}.jpg'), quality=85)
print(len(names), 'gambar ->', (len(names) + PER - 1) // PER, 'lembar di', out)
