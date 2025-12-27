"""
Script de Visualisation des Séquences de Gestes
Permet de visualiser les fichiers .npy enregistrés par capture_gestures.py
"""

import cv2
import numpy as np
from pathlib import Path
import time

class GestureVisualizer:
    def __init__(self, dataset_path="gesture_dataset"):
        """Initialise le visualiseur"""
        self.dataset_path = Path(dataset_path)
        self.sequences = []
        self.current_sequence_idx = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.playback_speed = 30  # FPS

        # Dimensions de l'image de visualisation
        self.img_width = 640
        self.img_height = 480

        # Connexions des landmarks de la main (même que HandTrackingModule)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        self.load_sequences()

    def load_sequences(self):
        """Charge toutes les séquences du dataset"""
        print("🔍 Chargement des séquences...")

        # Parcourir train et validation
        for split in ['train', 'validation']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue

            # Parcourir chaque dossier de geste
            for gesture_folder in split_path.iterdir():
                if not gesture_folder.is_dir():
                    continue

                gesture_name = gesture_folder.name

                # Charger tous les fichiers .npy
                for npy_file in gesture_folder.glob("*.npy"):
                    self.sequences.append({
                        'path': npy_file,
                        'gesture': gesture_name,
                        'split': split,
                        'filename': npy_file.name
                    })

        if len(self.sequences) == 0:
            print("❌ Aucune séquence trouvée dans le dataset !")
            print(f"   Vérifiez que le dossier '{self.dataset_path}' contient des fichiers .npy")
        else:
            print(f"✓ {len(self.sequences)} séquences chargées")

            # Afficher statistiques par geste
            gestures = {}
            for seq in self.sequences:
                gesture = seq['gesture']
                gestures[gesture] = gestures.get(gesture, 0) + 1

            print("\nRépartition par geste :")
            for gesture, count in sorted(gestures.items()):
                print(f"  - {gesture.ljust(15)}: {count} séquences")

    def get_current_sequence_data(self):
        """Charge les données de la séquence actuelle"""
        if len(self.sequences) == 0:
            return None

        seq_info = self.sequences[self.current_sequence_idx]
        data = np.load(seq_info['path'])
        return data, seq_info

    def draw_hand_landmarks(self, img, landmarks):
        """Dessine les landmarks de la main sur l'image"""
        h, w = img.shape[:2]

        # Normaliser les landmarks pour l'affichage
        # Les landmarks sont en coordonnées pixel, on les normalise puis les redimensionne
        landmarks_copy = landmarks.copy()

        # Trouver les limites pour normaliser
        x_coords = landmarks_copy[:, 0]
        y_coords = landmarks_copy[:, 1]

        # Centrer et mettre à l'échelle
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Éviter division par zéro
        x_range = max(x_max - x_min, 1)
        y_range = max(y_max - y_min, 1)

        # Normaliser entre 0 et 1
        x_norm = (x_coords - x_min) / x_range
        y_norm = (y_coords - y_min) / y_range

        # Mettre à l'échelle pour l'affichage (avec marges)
        margin = 50
        scale = min(w - 2*margin, h - 2*margin)

        x_display = (x_norm * scale + margin).astype(int)
        y_display = (y_norm * scale + margin).astype(int)

        # Dessiner les connexions
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (x_display[start_idx], y_display[start_idx])
                end_point = (x_display[end_idx], y_display[end_idx])
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)

        # Dessiner les landmarks
        for idx in range(len(landmarks)):
            cx, cy = x_display[idx], y_display[idx]

            # Couleur selon le type de landmark
            if idx == 0:  # Wrist
                color = (255, 0, 0)
                radius = 8
            elif idx in [4, 8, 12, 16, 20]:  # Fingertips
                color = (0, 0, 255)
                radius = 8
            else:
                color = (0, 255, 0)
                radius = 5

            cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), radius + 2, (255, 255, 255), 1)

    def draw_ui(self, img, seq_info, frame_idx, total_frames):
        """Dessine l'interface utilisateur"""
        # Fond semi-transparent pour le header
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.img_width, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Titre
        cv2.putText(img, "VISUALISEUR DE GESTES", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Informations sur la séquence
        gesture_text = f"Geste: {seq_info['gesture'].upper().replace('_', ' ')}"
        cv2.putText(img, gesture_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        split_text = f"Split: {seq_info['split'].upper()}"
        cv2.putText(img, split_text, (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        filename_text = f"Fichier: {seq_info['filename']}"
        cv2.putText(img, filename_text, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Numéro de séquence
        seq_counter = f"Sequence {self.current_sequence_idx + 1}/{len(self.sequences)}"
        cv2.putText(img, seq_counter, (self.img_width - 200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Statut lecture
        status = "▶ LECTURE" if self.is_playing else "⏸ PAUSE"
        status_color = (0, 255, 0) if self.is_playing else (0, 165, 255)
        cv2.putText(img, status, (self.img_width - 200, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Barre de progression
        progress_y = self.img_height - 80
        progress_bar_width = self.img_width - 40

        # Fond de la barre
        cv2.rectangle(img, (20, progress_y), (self.img_width - 20, progress_y + 20),
                     (50, 50, 50), -1)

        # Progression
        if total_frames > 0:
            progress = frame_idx / (total_frames - 1) if total_frames > 1 else 0
            progress_width = int(progress * progress_bar_width)
            cv2.rectangle(img, (20, progress_y), (20 + progress_width, progress_y + 20),
                         (0, 255, 0), -1)

        # Bordure de la barre
        cv2.rectangle(img, (20, progress_y), (self.img_width - 20, progress_y + 20),
                     (255, 255, 255), 2)

        # Frame counter
        frame_text = f"Frame: {frame_idx + 1}/{total_frames}"
        cv2.putText(img, frame_text, (20, progress_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Contrôles
        controls_y = self.img_height - 50
        cv2.putText(img, "Controles: [SPACE] Play/Pause  [←→] Frames  [↑↓] Sequences  [Q] Quitter",
                   (20, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return img

    def next_sequence(self):
        """Passe à la séquence suivante"""
        if len(self.sequences) > 0:
            self.current_sequence_idx = (self.current_sequence_idx + 1) % len(self.sequences)
            self.current_frame_idx = 0
            self.is_playing = False

    def previous_sequence(self):
        """Passe à la séquence précédente"""
        if len(self.sequences) > 0:
            self.current_sequence_idx = (self.current_sequence_idx - 1) % len(self.sequences)
            self.current_frame_idx = 0
            self.is_playing = False

    def next_frame(self, data):
        """Passe à la frame suivante"""
        if data is not None:
            self.current_frame_idx = min(self.current_frame_idx + 1, len(data) - 1)

    def previous_frame(self):
        """Passe à la frame précédente"""
        self.current_frame_idx = max(self.current_frame_idx - 1, 0)

    def toggle_playback(self):
        """Active/désactive la lecture automatique"""
        self.is_playing = not self.is_playing

    def run(self):
        """Lance la boucle principale de visualisation"""
        if len(self.sequences) == 0:
            print("\n❌ Aucune séquence à visualiser. Quittez avec Ctrl+C")
            return

        print("\n" + "=" * 70)
        print("CONTRÔLES :")
        print("  [SPACE] - Play/Pause")
        print("  [→] - Frame suivante")
        print("  [←] - Frame précédente")
        print("  [↑] - Séquence suivante")
        print("  [↓] - Séquence précédente")
        print("  [Q] - Quitter")
        print("=" * 70 + "\n")

        last_frame_time = time.time()
        frame_delay = 1.0 / self.playback_speed

        while True:
            # Charger la séquence actuelle
            result = self.get_current_sequence_data()
            if result is None:
                break

            data, seq_info = result

            # Créer l'image de visualisation
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

            # Dessiner les landmarks de la frame actuelle
            if self.current_frame_idx < len(data):
                landmarks = data[self.current_frame_idx]
                self.draw_hand_landmarks(img, landmarks)

            # Dessiner l'interface
            img = self.draw_ui(img, seq_info, self.current_frame_idx, len(data))

            # Afficher
            cv2.imshow("Visualiseur de Gestes", img)

            # Lecture automatique
            current_time = time.time()
            if self.is_playing and (current_time - last_frame_time) >= frame_delay:
                self.current_frame_idx += 1
                if self.current_frame_idx >= len(data):
                    self.current_frame_idx = 0  # Boucle
                last_frame_time = current_time

            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q ou ESC
                break
            elif key == ord(' '):  # SPACE
                self.toggle_playback()
            elif key == 81 or key == 2:  # Flèche gauche
                self.previous_frame()
                self.is_playing = False
            elif key == 83 or key == 3:  # Flèche droite
                self.next_frame(data)
                self.is_playing = False
            elif key == 82 or key == 0:  # Flèche haut
                self.next_sequence()
            elif key == 84 or key == 1:  # Flèche bas
                self.previous_sequence()

        cv2.destroyAllWindows()
        print("\n✓ Visualisation terminée\n")


def main():
    """Point d'entrée principal"""
    try:
        visualizer = GestureVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n\n→ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
