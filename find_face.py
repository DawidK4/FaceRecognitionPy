import cv2
import numpy as np
import uuid
import random

# Initialize Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load transparent PNG masks
masks = [cv2.imread(f'mask{i}.png', cv2.IMREAD_UNCHANGED) for i in range(1, 6)]

# Dictionary to track faces and assigned masks
tracked_faces = {}

selected_face_id = None  # Track which face is selected

def overlay_mask(frame, mask, x, y, w, h):
    mask_resized = cv2.resize(mask, (w, h))
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (
            frame[y:y+h, x:x+w, c] * (1 - mask_resized[:, :, 3] / 255.0)
            + mask_resized[:, :, c] * (mask_resized[:, :, 3] / 255.0)
        )

def select_face(event, mx, my, flags, param):
    global selected_face_id
    faces = param["faces"]
    ids = param["ids"]
    for i, (x, y, w, h) in enumerate(faces):
        if x <= mx <= x + w and y <= my <= y + h:
            selected_face_id = ids[i]
            return

def main():
    global selected_face_id
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Live Face Filter')
    faces_param = {"faces": [], "ids": []}
    cv2.setMouseCallback('Live Face Filter', select_face, faces_param)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        updated_tracked = {}
        face_ids = []
        faces_list = []

        for (x, y, w, h) in detected_faces:
            face_found = False
            for face_id, (prev_x, prev_y, mask_idx) in tracked_faces.items():
                if abs(prev_x - x) < 40 and abs(prev_y - y) < 40:
                    updated_tracked[face_id] = (x, y, mask_idx)
                    overlay_mask(frame, masks[mask_idx], x, y, w, h)
                    face_found = True
                    face_ids.append(face_id)
                    faces_list.append((x, y, w, h))
                    # Optionally, draw a green rectangle if selected
                    if face_id == selected_face_id:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break
            if not face_found:
                face_id = str(uuid.uuid4())
                mask_idx = random.randint(0, len(masks) - 1)
                updated_tracked[face_id] = (x, y, mask_idx)
                overlay_mask(frame, masks[mask_idx], x, y, w, h)
                face_ids.append(face_id)
                faces_list.append((x, y, w, h))
                if face_id == selected_face_id:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        faces_param["faces"] = faces_list
        faces_param["ids"] = face_ids

        tracked_faces.clear()
        tracked_faces.update(updated_tracked)

        cv2.imshow('Live Face Filter', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            if selected_face_id and selected_face_id in tracked_faces:
                x, y, _ = tracked_faces[selected_face_id]
                tracked_faces[selected_face_id] = (x, y, key - ord('1'))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
