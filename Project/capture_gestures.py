"""
Script de capture des landmarks pour le dataset
"""

import cv2
import numpy as np
from HandTrackingModule import handDetector
import time
import os
from pathlib import Path

class GestureDatasetCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = handDetector(mode='VIDEO', maxHands=1)

        self.fps = 30
        self.record_duration = 2.0  # secondes
        self.frames_per_gesture = int(self.fps * self.record_duration)

        self.gestures = [
            "idle",          # Main au repos
            "rotate_cw",     # Rotation horaire (Y)
            "rotate_ccw",    # Rotation anti-horaire (Y)
            "zoom_in",       # Agrandir (scale up)
            "zoom_out",      # Rétrécir (scale down)
            "swipe_left",    # Translation X-
            "swipe_right",   # Translation X+
            "swipe_up",      # Translation Y+
            "swipe_down"     # Translation Y-
]


        self.current_gesture_idx = 0
        self.current_gesture = self.gestures[self.current_gesture_idx]

        self.is_recording = False
        self.recorded_frames = []
        self.countdown = 0
        self.countdown_start = 0

        self.gesture_counts = {gesture: 0 for gesture in self.gestures}

        self.setup_directories()
        self.load_existing_counts()

    def setup_directories(self):
        self.base_path = Path("gesture_dataset")
        self.train_path = self.base_path / "train"
        self.val_path = self.base_path / "validation"

        for gesture in self.gestures:
            (self.train_path / gesture).mkdir(parents=True, exist_ok=True)
            (self.val_path / gesture).mkdir(parents=True, exist_ok=True)

        print(f"Folder created : {self.base_path.absolute()}")

    def load_existing_counts(self):
        for gesture in self.gestures:
            train_files = list((self.train_path / gesture).glob("*.npy"))
            val_files = list((self.val_path / gesture).glob("*.npy"))
            self.gesture_counts[gesture] = len(train_files) + len(val_files)

    def start_countdown(self):
        self.countdown = 3
        self.countdown_start = time.time()

    def update_countdown(self):
        if self.countdown > 0:
            elapsed = time.time() - self.countdown_start
            if elapsed >= 1.0:
                self.countdown -= 1
                self.countdown_start = time.time()
                if self.countdown == 0:
                    self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.recorded_frames = []
        self.record_start_time = time.time()


    def stop_recording(self):
        self.is_recording = False

        if len(self.recorded_frames) >= 10: 
            self.save_gesture()
        else:
            print("Short")

        self.recorded_frames = []

    def record_frame(self, landmarks):
        if landmarks and len(landmarks) == 21:
            frame_data = np.array([[lm[1], lm[2], 0.0] for lm in landmarks])
            self.recorded_frames.append(frame_data)

    def save_gesture(self):
        count = self.gesture_counts[self.current_gesture]
        is_validation = (count % 5 == 4)

        save_path = self.val_path if is_validation else self.train_path
        gesture_folder = save_path / self.current_gesture

        filename = f"{self.current_gesture}_{count:04d}.npy"
        filepath = gesture_folder / filename

        sequence = np.array(self.recorded_frames)
        np.save(filepath, sequence)

        self.gesture_counts[self.current_gesture] += 1

        split_name = "VALIDATION" if is_validation else "TRAIN"
        print(f"Saved : {filename} ({split_name}) - Shape: {sequence.shape}")
        print(f"Total : '{self.current_gesture}': {self.gesture_counts[self.current_gesture]} sequences")

    def next_gesture(self):
        self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
        self.current_gesture = self.gestures[self.current_gesture_idx]
        print(f"\n→ Next : {self.current_gesture.upper()}")

    def previous_gesture(self):
        self.current_gesture_idx = (self.current_gesture_idx - 1) % len(self.gestures)
        self.current_gesture = self.gestures[self.current_gesture_idx]
        print(f"\n← Prev : {self.current_gesture.upper()}")

    def draw_ui(self, img):
        h, w, _ = img.shape
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, "GESTURE", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        gesture_text = f"Gesture: {self.current_gesture.upper().replace('_', ' ')}"
        cv2.putText(img, gesture_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        count_text = f"Saved: {self.gesture_counts[self.current_gesture]}"
        cv2.putText(img, count_text, (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        gesture_progress = f"Gesture {self.current_gesture_idx + 1}/{len(self.gestures)}"
        cv2.putText(img, gesture_progress, (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        if self.countdown > 0:
            countdown_text = str(self.countdown)
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, 5, 10)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(img, countdown_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 255), 10)

        elif self.is_recording:
            cv2.circle(img, (w - 50, 50), 20, (0, 0, 255), -1)
            cv2.putText(img, "REC", (w - 90, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            frame_text = f"Frames: {len(self.recorded_frames)}/{self.frames_per_gesture}"
            cv2.putText(img, frame_text, (w - 200, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        controls_y = h - 150
        cv2.putText(img, "r = Record", (20, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "n = Next gesture", (20, controls_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "p = Previous gesture", (20, controls_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "q = Quit", (20, controls_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def run(self):
        print("\nCONTRÔLES :")
        print("  r = Record (3 second countdown)")
        print("  n = Next gesture")
        print("  p = Previous gesture")
        print("  q = Quit")
        print(f"\n→ Geste actuel : {self.current_gesture.upper()}\n")

        while True:
            success, img = self.cap.read()
            if not success:
                print("Erreur : Impossible de lire la caméra")
                break
            img = cv2.flip(img, 1)
            img = self.detector.findHands(img, draw=True)
            lmList, bbox = self.detector.findPosition(img, draw=False)

            if self.is_recording:
                self.record_frame(lmList)

                if len(self.recorded_frames) >= self.frames_per_gesture:
                    self.stop_recording()

            if self.countdown > 0:
                self.update_countdown()

            img = self.draw_ui(img)

            cv2.imshow("Dataset Capture", img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
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

        total = 0
        for gesture in self.gestures:
            count = self.gesture_counts[gesture]
            total += count
            print(f"  {gesture.ljust(15)}: {count} sequences")

        print(f"  TOTAL: {total}")
        print(f"\nSave in : {self.base_path.absolute()}")


def main():
    """Point d'entrée principal"""
    try:
        capturer = GestureDatasetCapture()
        capturer.run()
    except Exception as e:
        print(f"\nErreur : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
