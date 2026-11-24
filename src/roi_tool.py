import cv2
import json
import argparse
import os

polygons = [[]]
current_poly = 0
img_display = None
img_original = None

def mouse_callback(event, x, y, flags, param):
    global polygons, current_poly, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        polygons[current_poly].append([x, y])
        draw_polygons()

def draw_polygons():
    global img_display, img_original, polygons
    img_display = img_original.copy()
    for poly in polygons:
        if len(poly) > 0:
            for i in range(len(poly)):
                cv2.circle(img_display, tuple(poly[i]), 4, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(img_display, tuple(poly[i-1]), tuple(poly[i]), (0, 255, 0), 2)
            if len(poly) > 2:
                cv2.line(img_display, tuple(poly[-1]), tuple(poly[0]), (0, 255, 0), 2)
    cv2.imshow("ROI Tool", img_display)

def main():
    global img_original, polygons, current_poly
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--verify', action='store_true') # Tambahan opsi verify
    args = parser.parse_args()

    img_original = cv2.imread(args.image)
    if img_original is None:
        print("Gambar tidak ditemukan! Cek kembali path --image.")
        return

    h, w = img_original.shape[:2]
    out_path = f"config/roi/{args.camera}.json"

    # Fitur memuat JSON jika --verify diaktifkan
    if args.verify:
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                data = json.load(f)
                polygons = data.get("polygons", [[]])
            print(f"Memuat ROI dari {out_path} untuk verifikasi.")
        else:
            print(f"File {out_path} belum ada!")

    cv2.namedWindow("ROI Tool")
    cv2.setMouseCallback("ROI Tool", mouse_callback)
    draw_polygons()

    print("Kontrol: Klik Kiri (Tambah titik), 'u' (Undo), 'n' (Poligon Baru), 's' (Simpan), 'q' (Keluar)")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            if len(polygons[current_poly]) > 0:
                polygons[current_poly].pop()
            elif current_poly > 0:
                polygons.pop()
                current_poly -= 1
            draw_polygons()
        elif key == ord('n'):
            if len(polygons[current_poly]) > 0:
                polygons.append([])
                current_poly += 1
        elif key == ord('s'):
            os.makedirs("config/roi", exist_ok=True)
            valid_polys = [p for p in polygons if len(p) > 2]
            with open(out_path, 'w') as f:
                json.dump({"frame_size": [w, h], "polygons": valid_polys}, f)
            print(f"[OK] ROI tersimpan di {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()