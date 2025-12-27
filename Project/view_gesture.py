"""
Script simple de visualisation d'un fichier de geste .npy
Usage: python view_gesture.py chemin/vers/fichier.npy
"""
#../gesture_dataset/train/zoom_in/zoom_in_xxxx.npy

import cv2
import numpy as np
import sys
import time

# Connexions des landmarks de la main
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)            # Palm
]

def draw_hand(img, landmarks):
    """Dessine la main sur l'image"""
    h, w = img.shape[:2]

    # Normaliser les landmarks
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    # Normaliser et mettre à l'échelle
    margin = 50
    scale = min(w - 2*margin, h - 2*margin)

    x_norm = (x_coords - x_min) / x_range
    y_norm = (y_coords - y_min) / y_range

    x_display = (x_norm * scale + margin).astype(int)
    y_display = (y_norm * scale + margin).astype(int)

    # Dessiner les connexions
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_point = (x_display[start_idx], y_display[start_idx])
        end_point = (x_display[end_idx], y_display[end_idx])
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)

    # Dessiner les landmarks
    for idx in range(len(landmarks)):
        cx, cy = x_display[idx], y_display[idx]

        if idx == 0:  # Poignet
            color = (255, 0, 0)
            radius = 8
        elif idx in [4, 8, 12, 16, 20]:  # Bouts des doigts
            color = (0, 0, 255)
            radius = 8
        else:
            color = (0, 255, 0)
            radius = 5

        cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)

def main():
    if len(sys.argv) != 2:
        print("python view_gesture.py <fichier.npy>")
        print("Exemple: python view_gesture.py gesture_dataset/train/rotate_cw/rotate_cw_0001.npy")
        sys.exit(1)

    filepath = sys.argv[1]

    # Charger le fichier
    try:
        data = np.load(filepath)
        print(f"✓ Fichier chargé: {filepath}")
        print(f"  Shape: {data.shape}")
        print(f"  Frames: {len(data)}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        sys.exit(1)

    print("\nContrôles:")
    print("  [SPACE] - Pause/Play")
    print("  [Q] - Quitter\n")

    # Visualisation
    img_width, img_height = 640, 480
    frame_idx = 0
    is_playing = True
    fps = 30

    while True:
        # Créer image
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Dessiner la main
        draw_hand(img, data[frame_idx])

        # Info frame
        info_text = f"Frame: {frame_idx + 1}/{len(data)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status = "Playing" if is_playing else "Paused"
        cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_playing else (0, 165, 255), 2)

        # Barre de progression
        progress = frame_idx / (len(data) - 1) if len(data) > 1 else 0
        bar_width = int(progress * (img_width - 20))
        cv2.rectangle(img, (10, img_height - 30), (10 + bar_width, img_height - 10), (0, 255, 0), -1)
        cv2.rectangle(img, (10, img_height - 30), (img_width - 10, img_height - 10), (255, 255, 255), 2)

        cv2.imshow("Gesture Viewer", img)

        # Avancer si en lecture
        if is_playing:
            frame_idx = (frame_idx + 1) % len(data)
            time.sleep(1.0 / fps)

        # Gestion touches
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            is_playing = not is_playing

    cv2.destroyAllWindows()
    print("\n✓ Visualisation terminée")

if __name__ == "__main__":
    main()
