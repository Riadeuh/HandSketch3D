"""
Script de Capture de Dataset pour Gestes
Enregistre des séquences de landmarks pour entraîner un modèle de reconnaissance de gestes
"""

import cv2
import numpy as np
from HandTrackingModule import handDetector
import time
import os
from pathlib import Path

class GestureDatasetCapture:
    def __init__(self):
        """Initialise le système de capture"""
        self.cap = cv2.VideoCapture(0)
        self.detector = handDetector(mode='VIDEO', maxHands=1)

        # Configuration
        self.fps = 30
        self.record_duration = 2.0  # secondes
        self.frames_per_gesture = int(self.fps * self.record_duration)

        # Liste des gestes à enregistrer
        self.gestures = [
            "idle",          # Main au repos
            "rotate_cw",     # Rotation horaire
            "rotate_ccw",    # Rotation anti-horaire
            "zoom_in",       # Pincement qui s'écarte
            "zoom_out",      # Main qui se referme
            "swipe_left",    # Balayage gauche
            "swipe_right",   # Balayage droite
            "push",          # Mouvement vers l'avant
            "pull",          # Mouvement vers soi
            "grab",          # Main ouverte → fermée
            "release"        # Main fermée → ouverte
        ]

        self.current_gesture_idx = 0
        self.current_gesture = self.gestures[self.current_gesture_idx]

        # État de l'enregistrement
        self.is_recording = False
        self.recorded_frames = []
        self.countdown = 0
        self.countdown_start = 0

        # Statistiques
        self.gesture_counts = {gesture: 0 for gesture in self.gestures}

        # Créer la structure de dossiers
        self.setup_directories()
        self.load_existing_counts()

    def setup_directories(self):
        """Crée les dossiers pour sauvegarder le dataset"""
        self.base_path = Path("gesture_dataset")
        self.train_path = self.base_path / "train"
        self.val_path = self.base_path / "validation"

        for gesture in self.gestures:
            (self.train_path / gesture).mkdir(parents=True, exist_ok=True)
            (self.val_path / gesture).mkdir(parents=True, exist_ok=True)

        print(f"✓ Dossiers créés dans : {self.base_path.absolute()}")

    def load_existing_counts(self):
        """Compte les fichiers déjà enregistrés"""
        for gesture in self.gestures:
            train_files = list((self.train_path / gesture).glob("*.npy"))
            val_files = list((self.val_path / gesture).glob("*.npy"))
            self.gesture_counts[gesture] = len(train_files) + len(val_files)

    def start_countdown(self):
        """Démarre le compte à rebours avant l'enregistrement"""
        self.countdown = 3
        self.countdown_start = time.time()

    def update_countdown(self):
        """Met à jour le compte à rebours"""
        if self.countdown > 0:
            elapsed = time.time() - self.countdown_start
            if elapsed >= 1.0:
                self.countdown -= 1
                self.countdown_start = time.time()
                if self.countdown == 0:
                    self.start_recording()

    def start_recording(self):
        """Démarre l'enregistrement des landmarks"""
        self.is_recording = True
        self.recorded_frames = []
        self.record_start_time = time.time()
        print(f"🔴 ENREGISTREMENT : {self.current_gesture}")

    def stop_recording(self):
        """Arrête l'enregistrement et sauvegarde"""
        self.is_recording = False

        if len(self.recorded_frames) >= 10:  # Minimum de frames
            self.save_gesture()
        else:
            print("❌ Séquence trop courte, ignorée")

        self.recorded_frames = []

    def record_frame(self, landmarks):
        """Enregistre les landmarks d'une frame"""
        if landmarks and len(landmarks) == 21:
            # Extraire seulement les coordonnées x, y, z
            frame_data = np.array([[lm[1], lm[2], 0.0] for lm in landmarks])
            self.recorded_frames.append(frame_data)

    def save_gesture(self):
        """Sauvegarde la séquence de geste"""
        # Décider si train ou validation (80/20 split)
        count = self.gesture_counts[self.current_gesture]
        is_validation = (count % 5 == 4)  # Chaque 5ème exemple va dans validation

        save_path = self.val_path if is_validation else self.train_path
        gesture_folder = save_path / self.current_gesture

        # Nom du fichier
        filename = f"{self.current_gesture}_{count:04d}.npy"
        filepath = gesture_folder / filename

        # Convertir en numpy array et sauvegarder
        sequence = np.array(self.recorded_frames)
        np.save(filepath, sequence)

        # Mettre à jour les compteurs
        self.gesture_counts[self.current_gesture] += 1

        split_name = "VALIDATION" if is_validation else "TRAIN"
        print(f"✓ Sauvegardé : {filename} ({split_name}) - Shape: {sequence.shape}")
        print(f"  Total pour '{self.current_gesture}': {self.gesture_counts[self.current_gesture]} séquences")

    def next_gesture(self):
        """Passe au geste suivant"""
        self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
        self.current_gesture = self.gestures[self.current_gesture_idx]
        print(f"\n→ Geste suivant : {self.current_gesture.upper()}")

    def previous_gesture(self):
        """Retourne au geste précédent"""
        self.current_gesture_idx = (self.current_gesture_idx - 1) % len(self.gestures)
        self.current_gesture = self.gestures[self.current_gesture_idx]
        print(f"\n← Geste précédent : {self.current_gesture.upper()}")

    def draw_ui(self, img):
        """Dessine l'interface utilisateur sur l'image"""
        h, w, _ = img.shape

        # Fond semi-transparent pour le texte
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Titre
        cv2.putText(img, "CAPTURE DE DATASET - GESTES", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Geste actuel
        gesture_text = f"Geste: {self.current_gesture.upper().replace('_', ' ')}"
        cv2.putText(img, gesture_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Compteur
        count_text = f"Enregistres: {self.gesture_counts[self.current_gesture]}"
        cv2.putText(img, count_text, (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Progression geste
        gesture_progress = f"Geste {self.current_gesture_idx + 1}/{len(self.gestures)}"
        cv2.putText(img, gesture_progress, (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Instructions
        cv2.putText(img, "Controles:", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # État de l'enregistrement
        if self.countdown > 0:
            # Compte à rebours
            countdown_text = str(self.countdown)
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_BOLD, 5, 10)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(img, countdown_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 255), 10)

        elif self.is_recording:
            # Indicateur d'enregistrement
            cv2.circle(img, (w - 50, 50), 20, (0, 0, 255), -1)
            cv2.putText(img, "REC", (w - 90, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            # Barre de progression
            progress = len(self.recorded_frames) / self.frames_per_gesture
            bar_width = int(progress * (w - 40))
            cv2.rectangle(img, (20, h - 40), (20 + bar_width, h - 20), (0, 0, 255), -1)
            cv2.rectangle(img, (20, h - 40), (w - 20, h - 20), (255, 255, 255), 2)

            # Nombre de frames
            frame_text = f"Frames: {len(self.recorded_frames)}/{self.frames_per_gesture}"
            cv2.putText(img, frame_text, (w - 200, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Contrôles en bas à gauche
        controls_y = h - 150
        cv2.putText(img, "[R] Enregistrer", (20, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "[N] Geste suivant", (20, controls_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "[P] Geste precedent", (20, controls_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "[Q] Quitter", (20, controls_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def run(self):
        """Lance la boucle principale de capture"""
        print("\n" + "=" * 70)
        print("CAPTURE DE DATASET POUR RECONNAISSANCE DE GESTES")
        print("=" * 70)
        print("\nGestes à enregistrer :")
        for i, gesture in enumerate(self.gestures, 1):
            count = self.gesture_counts[gesture]
            print(f"  {i}. {gesture.ljust(15)} - {count} séquences")
        print("\n" + "=" * 70)
        print("\nCONTRÔLES :")
        print("  [R] - Démarrer l'enregistrement (compte à rebours 3s)")
        print("  [N] - Passer au geste suivant")
        print("  [P] - Revenir au geste précédent")
        print("  [Q] - Quitter et sauvegarder")
        print("=" * 70)
        print(f"\n→ Geste actuel : {self.current_gesture.upper()}\n")

        while True:
            success, img = self.cap.read()
            if not success:
                print("❌ Erreur : Impossible de lire la caméra")
                break

            img = cv2.flip(img, 1)

            # Détection de la main
            img = self.detector.findHands(img, draw=True)
            lmList, bbox = self.detector.findPosition(img, draw=False)

            # Enregistrement si actif
            if self.is_recording:
                self.record_frame(lmList)

                # Arrêter si durée atteinte
                if len(self.recorded_frames) >= self.frames_per_gesture:
                    self.stop_recording()

            # Mise à jour du compte à rebours
            if self.countdown > 0:
                self.update_countdown()

            # Dessiner l'interface
            img = self.draw_ui(img)

            cv2.imshow("Capture de Dataset", img)

            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n→ Arrêt de la capture...")
                break

            elif key == ord('r') and not self.is_recording and self.countdown == 0:
                self.start_countdown()

            elif key == ord('n') and not self.is_recording:
                self.next_gesture()

            elif key == ord('p') and not self.is_recording:
                self.previous_gesture()

        self.cleanup()

    def cleanup(self):
        """Nettoie et affiche les statistiques finales"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

        print("\n" + "=" * 70)
        print("STATISTIQUES FINALES")
        print("=" * 70)

        total = 0
        for gesture in self.gestures:
            count = self.gesture_counts[gesture]
            total += count
            print(f"  {gesture.ljust(15)}: {count} séquences")

        print("-" * 70)
        print(f"  TOTAL: {total} séquences enregistrées")
        print("=" * 70)
        print(f"\n✓ Dataset sauvegardé dans : {self.base_path.absolute()}")
        print("✓ Prêt pour l'entraînement du modèle !\n")


def main():
    """Point d'entrée principal"""
    try:
        capturer = GestureDatasetCapture()
        capturer.run()
    except KeyboardInterrupt:
        print("\n\n→ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
