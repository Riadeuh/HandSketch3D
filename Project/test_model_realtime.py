"""
Script de test du modèle CNN 1D en temps réel
Affiche le geste reconnu depuis la webcam
"""

import cv2
import numpy as np
from HandTrackingModule import handDetector
import json
from pathlib import Path
import time

try:
    from tensorflow import keras
except ImportError:
    print("Error: TensorFlow not installed")
    print("Install: pip install tensorflow")
    exit(1)


class GestureRecognizer:
    def __init__(self, model_path="Models/best_model.keras", metadata_path="Models/metadata.json"):
        """Charge le modèle et les métadonnées"""

        # Charger métadonnées
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.classes = metadata['classes']
        self.input_shape = metadata['input_shape']  # [60, 63]
        self.num_frames = 20  # 60 frames

        # Charger le modèle
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Classes: {self.classes}")
        print(f"✓ Input shape: {self.input_shape}")

        # Buffer pour stocker les frames
        self.frame_buffer = []

        # Détecteur de main
        self.detector = handDetector(mode='VIDEO', maxHands=1)

    def extract_features(self, landmarks):
        """Extrait les features depuis les landmarks (21 points × 3 coords = 63)"""
        if landmarks and len(landmarks) == 21:
            # landmarks format: [[id, x, y], ...]
            # On prend seulement x, y, z=0
            features = np.array([[lm[1], lm[2], 0.0] for lm in landmarks])
            return features.flatten()  # Shape: (63,)
        return None

    def normalize_sequence(self, sequence):
        """
        Normalise une séquence de landmarks comme à l'entraînement
        Args:
            sequence: np.array de shape (60, 63)
        Returns:
            sequence normalisée
        """
        # Reshape en (60, 21, 3)
        landmarks = sequence.reshape(self.num_frames, 21, 3)

        # Centrer par rapport au poignet (landmark 0)
        wrist = landmarks[:, 0:1, :]  # Shape: (60, 1, 3)
        landmarks_centered = landmarks - wrist

        # Normaliser par la taille de la paume (distance poignet-majeur)
        palm_size = np.linalg.norm(landmarks[:, 9, :] - landmarks[:, 0, :], axis=1)  # Shape: (60,)
        palm_size = palm_size[:, np.newaxis, np.newaxis]  # Shape: (60, 1, 1)
        palm_size = np.where(palm_size == 0, 1, palm_size)  # Éviter division par zéro

        landmarks_norm = landmarks_centered / palm_size

        # Reshape back en (60, 63)
        return landmarks_norm.reshape(self.num_frames, -1)

    def predict(self):
        """Fait une prédiction si on a assez de frames"""
        if len(self.frame_buffer) < self.num_frames:
            return None, 0.0

        # Prendre les 60 dernières frames
        sequence = np.array(self.frame_buffer[-self.num_frames:])  # Shape: (60, 63)

        # IMPORTANT: Normaliser comme à l'entraînement
        sequence = self.normalize_sequence(sequence)  # Shape: (60, 63)

        # Ajouter dimension batch
        sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 60, 63)

        # Prédiction
        predictions = self.model.predict(sequence, verbose=0)

        # Classe prédite et confiance
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        gesture = self.classes[predicted_class_idx]

        return gesture, confidence

    def run(self):
        """Lance la reconnaissance en temps réel"""
        cap = cv2.VideoCapture(0)

        print("\nControls:")
        print("  [Q] - Quit")
        print("\nStarting recognition...\n")

        current_gesture = "Waiting..."
        confidence = 0.0

        pTime = 0
        cTime = 0

        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)

            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime

            # Détection main
            img = self.detector.findHands(img, draw=True)
            lmList, bbox = self.detector.findPosition(img, draw=False)

            # Extraire features
            features = self.extract_features(lmList)

            if features is not None:
                # Ajouter au buffer
                self.frame_buffer.append(features)

                # Limiter taille du buffer (garder 60 dernières frames)
                if len(self.frame_buffer) > self.num_frames:
                    self.frame_buffer.pop(0)

                # Prédire chaque frame pour plus de réactivité
                if len(self.frame_buffer) >= self.num_frames:
                    gesture, conf = self.predict()
                    if gesture:
                        current_gesture = gesture
                        confidence = conf
            else:
                # Pas de main détectée, vider le buffer
                self.frame_buffer = []
                current_gesture = "No hand"
                confidence = 0.0

            # Affichage
            h, w, _ = img.shape

            # Fond semi-transparent
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # Geste détecté
            gesture_text = f"Gesture: {current_gesture.upper().replace('_', ' ')}"
            cv2.putText(img, gesture_text, (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Confiance
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(img, conf_text, (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Buffer status
            buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.num_frames}"
            cv2.putText(img, buffer_text, (w - 220, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(img, fps_text, (w - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Barre de confiance
            if confidence > 0:
                bar_width = int(confidence * (w - 40))
                cv2.rectangle(img, (20, 100), (20 + bar_width, 115), (0, 255, 0), -1)
            cv2.rectangle(img, (20, 100), (w - 20, 115), (255, 255, 255), 2)

            cv2.imshow("Gesture Recognition - Real Time", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("\n✓ Recognition stopped")


def main():
    try:
        recognizer = GestureRecognizer()
        recognizer.run()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure the model files exist:")
        print("  - Models/best_model.keras")
        print("  - Models/metadata.json")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
