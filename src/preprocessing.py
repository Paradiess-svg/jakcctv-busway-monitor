"""preprocessing.py -- Tahap 2 (kode Ivan, 7 Nov; Dihar/Claude 17 Nov: siang/malam dari jam). Contoh: python src/preprocessing.py --image x.jpg --jam 19
"""
import cv2
import numpy as np
import argparse
import yaml
import os
import glob

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def malam_dari_jam(jam, cfg):
    """jam = angka (19 / 5.5) atau string 'HH', 'HH:MM', 'HH:MM:SS'. Malam = >= malam_mulai_jam atau < malam_selesai_jam."""
    p = cfg.get('preprocessing', {})
    if isinstance(jam, str):
        bagian = [int(x) for x in jam.split(':')]
        h = bagian[0] + (bagian[1] / 60 if len(bagian) > 1 else 0)
    else:
        h = float(jam)
    return h >= p.get('malam_mulai_jam', 18) or h < p.get('malam_selesai_jam', 5.5)

def detect_day_night(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return 'night' if avg_brightness < threshold else 'day'

def apply_clahe(img, clip_limit, tile):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def apply_gamma(img, gamma):
    lut = ((np.arange(256) / 255.0) ** (1 / gamma) * 255).astype('uint8')
    return cv2.LUT(img, lut)

def apply_denoise(img, method):
    if method == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif method == 'fastNlMeans':
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img

def suppress_glare(img, threshold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = v > threshold
    # Turunkan kecerahan pada area yang terlalu silau (glare)
    v[mask] = np.maximum(v[mask] * 0.8, threshold).astype(np.uint8) 
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def apply_unsharp(img, amount):
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)

def process(img, cfg, jam=None):
    """jam (opsional) = jam pengambilan gambar; kalau ada, siang/malam ditentukan dari jam (lebih andal)."""
    prep_cfg = cfg.get('preprocessing', {})
    
    # 1. Deteksi Siang/Malam
    if jam is not None:
        time_of_day = 'night' if malam_dari_jam(jam, cfg) else 'day'
    else:
        time_of_day = detect_day_night(img, prep_cfg.get('day_night_threshold', 70))
    result = img.copy()
    
    # 2. Suppress Glare (Opsional)
    glare_cfg = prep_cfg.get('glare', {})
    if glare_cfg.get('enabled', False):
        result = suppress_glare(result, glare_cfg.get('threshold', 240))
        
    # 3. CLAHE
    clahe_cfg = prep_cfg.get('clahe', {})
    if clahe_cfg.get('enabled', False):
        result = apply_clahe(result, clahe_cfg.get('clip_limit', 2.0), clahe_cfg.get('tile', 8))
        
    # 4. Gamma (KHUSUS MALAM)
    gamma_cfg = prep_cfg.get('gamma_night', {})
    if gamma_cfg.get('enabled', False) and time_of_day == 'night':
        result = apply_gamma(result, gamma_cfg.get('value', 1.5))
        
    # 5. Denoise
    denoise_cfg = prep_cfg.get('denoise', {})
    if denoise_cfg.get('enabled', False):
        result = apply_denoise(result, denoise_cfg.get('method', 'bilateral'))
        
    # 6. Unsharp (Opsional)
    unsharp_cfg = prep_cfg.get('unsharp', {})
    if unsharp_cfg.get('enabled', False):
        result = apply_unsharp(result, unsharp_cfg.get('amount', 0.5))
        
    return result, time_of_day

def main():
    parser = argparse.ArgumentParser(description="Script Preprocessing Gambar TransJakarta (Otomatis Siang/Malam)")
    parser.add_argument("--image", help="Path ke satu gambar input")
    parser.add_argument("--out", default="outputs/hasil_preprocessing.jpg", help="Path untuk menyimpan hasil 1 gambar")
    parser.add_argument("--in-dir", help="Path ke folder berisi banyak gambar (Batch processing)")
    parser.add_argument("--out-dir", help="Path folder untuk menyimpan hasil batch")
    parser.add_argument("--jam", help="jam pengambilan (HH atau HH:MM:SS) untuk menentukan siang/malam")
    args = parser.parse_args()

    config = load_config()

    # Mode 1: Batch Directory Processing
    if args.in_dir and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        # Cari file jpg dan png
        image_files = glob.glob(os.path.join(args.in_dir, '*.[jp][pn]*[g]')) 
        print(f"Ditemukan {len(image_files)} gambar di {args.in_dir}")
        
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                hasil, waktu = process(img, config)
                nama_file = os.path.basename(img_path)
                out_path = os.path.join(args.out_dir, f"{waktu}_{nama_file}")
                cv2.imwrite(out_path, hasil)
        print(f"[OK] Selesai memproses folder. Hasil di {args.out_dir}")

    # Mode 2: Single Image Processing
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Gambar {args.image} tidak ditemukan!")
            return
        
        hasil, waktu = process(img, config, args.jam)
        cv2.imwrite(args.out, hasil)
        print(f"[OK] Gambar terdeteksi sebagai {waktu.upper()}. Hasil disimpan di {args.out}")
        
    else:
        print("Error: Gunakan --image dan --out ATAU --in-dir dan --out-dir. Ketik --help untuk bantuan.")

if __name__ == "__main__":
    main()