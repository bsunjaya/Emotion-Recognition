import cv2
from deepface import DeepFace
import time, sys

print("[INFO] Loading face detection model...")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    sys.exit("[ERROR] Cannot load 'haarcascade_frontalface_default.xml'. Put it in this folder or fix the path.")
print("[INFO] Face detection model loaded.")

print("[INFO] Accessing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    sys.exit("[ERROR] Camera not available. Check permissions or device index.")
print("[INFO] Camera ready.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] DeepFace will lazy-load the emotion model on first analysis...")

ANALYZE_EVERY = 1   # try every frame for now; you can raise to 3–5 later
MIN_FACE = 60
frame_idx = 0

print("[INFO] Starting real-time emotion detection. Press 'q' to quit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Frame capture failed, skipping...")
            time.sleep(0.02)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"[DEBUG] Frame {frame_idx}: {len(faces)} face(s)")

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            if w < MIN_FACE or h < MIN_FACE:
                continue

            if frame_idx % ANALYZE_EVERY == 0:
                face_bgr = frame[y:y+h, x:x+w]
                try:
                    # Keep it simple for compatibility: no prog_bar param
                    # Also skip re-detection since we already have the ROI
                    res = DeepFace.analyze(
                        face_bgr,
                        actions=['emotion'],
                        detector_backend='skip',   # if this errors, remove this line
                        enforce_detection=False
                    )
                    data = res[0] if isinstance(res, list) else res

                    # Handle both possible response shapes
                    emotion = (
                        data.get('dominant_emotion')
                        or (data.get('emotion') or {}).get('dominant_emotion')
                    )

                    if emotion:
                        print(f"[RESULT] Face {i+1}: {emotion}")
                        cv2.putText(frame, emotion, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except TypeError as e:
                    # If detector_backend='skip' isn't supported in your version
                    if "detector_backend" in str(e):
                        try:
                            res = DeepFace.analyze(
                                face_bgr,
                                actions=['emotion'],
                                enforce_detection=False
                            )
                            data = res[0] if isinstance(res, list) else res
                            emotion = (
                                data.get('dominant_emotion')
                                or (data.get('emotion') or {}).get('dominant_emotion')
                            )
                            if emotion:
                                print(f"[RESULT] Face {i+1}: {emotion}")
                                cv2.putText(frame, emotion, (x, y-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        except Exception as e2:
                            print(f"[ERROR] Analysis retry failed for face {i+1}: {e2}")
                    else:
                        print(f"[ERROR] Analysis failed for face {i+1}: {e}")
                except Exception as e:
                    print(f"[ERROR] Analysis failed for face {i+1}: {e}")

        cv2.imshow('Real-time Emotion Detection', frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Exiting...")
finally:    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released and windows closed.")
