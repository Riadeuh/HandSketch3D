import cv2
from HandTrackingModule import handDetector
import numpy as np
import time

def main():
    """Main function to test the algo"""
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    
    detector = handDetector(mode='VIDEO', maxHands=2)
    
    canvas = None
    
    print("=" * 60)
    print("Hand Detector with Drawing Mode initialized")
    print("=" * 60)
    print("CONTROLS:")
    print("  - Index finger up alone: DRAW")
    print("  - All fingers up: CLEAR drawing")
    print("  - 'q': Quit")
    print("  - 'c': Clear canvas manually")
    print("  - '1': Red color")
    print("  - '2': Green color")
    print("  - '3': Blue color")
    print("=" * 60)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        img = cv2.flip(img, 1)
        
        if canvas is None:
            canvas = np.zeros_like(img)
        
        img = detector.findHands(img, draw=True)
        lmList, bbox = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            index_tip = detector.getIndexFingerTip()
            if detector.checkDrawingGesture():
                if not detector.is_drawing:
                    detector.startDrawing()   
                if index_tip:
                    detector.addDrawPoint(index_tip)
                    cv2.circle(img, index_tip, 10, detector.draw_color, cv2.FILLED)
            else:
                if detector.is_drawing:
                    detector.stopDrawing()
            if detector.checkEraseGesture():
                detector.clearDrawing()
                canvas = np.zeros_like(img)
                cv2.putText(img, "CLEARED", (250, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        canvas = detector.drawOnCanvas(canvas)
        
        img_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        
        status = "DRAWING" if detector.is_drawing else "NORMAL"
        status_color = (0, 255, 0) if detector.is_drawing else (0, 0, 255)
        cv2.putText(img_combined, f"Status: {status}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.putText(img_combined, f"Points: {len(detector.draw_points)}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img_combined, f'FPS: {int(fps)}', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Hand Drawing", img_combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.clearDrawing()
            canvas = np.zeros_like(img)
        elif key == ord('1'):
            detector.setDrawColor((0, 0, 255))  # Red
            print("Color: RED")
        elif key == ord('2'):
            detector.setDrawColor((0, 255, 0))  # Green
            print("Color: GREEN")
        elif key == ord('3'):
            detector.setDrawColor((255, 0, 0))  # Blue
            print("Color: BLUE")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()