# import cv2
# from ultralytics import YOLO
# import numpy as np

# # Load YOLOv8 model (YOLOv8n is lightweight)
# model = YOLO('yolov8n.pt')  # You can use yolov8s.pt or yolov8m.pt for better accuracy

# # Define item and bag classes
# SUSPICIOUS_CLASSES = ['bottle', 'handbag', 'backpack', 'cart', 'person']

# # Video path
# video_path = 'test_video.mp4'
# cap = cv2.VideoCapture(video_path)

# frame_count = 0
# suspicious_flag = False

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_count += 1

#     # Run YOLO inference
#     results = model(frame)[0]
#     boxes = results.boxes
#     cls_names = results.names
#     detections = []

#     for box in boxes:
#         cls_id = int(box.cls[0])
#         cls_name = cls_names[cls_id]
#         if cls_name in SUSPICIOUS_CLASSES:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             detections.append((cls_name, (x1, y1, x2, y2), conf))
#             color = (0, 255, 0)
#             label = f"{cls_name} {conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Simple logic: if item is near bag but not cart, mark suspicious
#     item_boxes = [box for cls, box, _ in detections if cls == 'bottle']
#     bag_boxes = [box for cls, box, _ in detections if cls in ['handbag', 'backpack']]
#     cart_boxes = [box for cls, box, _ in detections if cls == 'cart']

#     for item_box in item_boxes:
#         ix1, iy1, ix2, iy2 = item_box
#         item_center = ((ix1 + ix2) // 2, (iy1 + iy2) // 2)

#         for bag_box in bag_boxes:
#             bx1, by1, bx2, by2 = bag_box
#             bag_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
#             distance = np.linalg.norm(np.array(item_center) - np.array(bag_center))
#             if distance < 80:  # suspiciously close
#                 suspicious_flag = True
#                 cv2.putText(frame, "⚠️ Suspicious Behavior Detected!", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#     cv2.imshow("Shoplifting Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# shoplifting_tracker.py

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for more accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Class labels
BAG_CLASSES = ['handbag', 'backpack']
PERSON_CLASS = 'person'
CART_CLASS = 'cart'

# Load your test video
cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes = results.boxes
    class_names = results.names

    dets = []
    objects = []

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))
        objects.append((cls_name, (x1, y1, x2, y2)))

    tracks = tracker.update_tracks(dets, frame=frame)

    person_positions = []
    item_positions = []
    bag_positions = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        cls_name = track.get_det_class()

        x1, y1, x2, y2 = map(int, ltrb)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if cls_name == PERSON_CLASS:
            person_positions.append((track_id, (x1, y1, x2, y2), center))
            color = (255, 255, 0)
        elif cls_name in BAG_CLASSES:
            bag_positions.append((track_id, (x1, y1, x2, y2), center))
            color = (255, 0, 255)
        elif cls_name != CART_CLASS:
            item_positions.append((track_id, (x1, y1, x2, y2), center))
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        label = f"{cls_name} ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Suspicion logic
    for item_id, item_box, item_center in item_positions:
        for person_id, person_box, person_center in person_positions:
            px1, py1, px2, py2 = person_box
            pocket_area = (py1 + 2 * (py2 - py1) // 3)

            if py1 < item_center[1] < py2 and item_center[1] > pocket_area:
                cv2.putText(frame, "⚠️ Suspicious: Into Pocket", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        for bag_id, bag_box, bag_center in bag_positions:
            distance = np.linalg.norm(np.array(item_center) - np.array(bag_center))
            if distance < 120:
                cv2.putText(frame, "⚠️ Suspicious: Into Bag", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Shoplifting Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from collections import defaultdict

# # Load YOLOv8 model
# model = YOLO('yolov8n.pt')
# tracker = DeepSort(max_age=30)

# # Track item movements
# item_history = defaultdict(list)

# # Class labels
# PERSON_CLASS = 'person'
# BAG_CLASSES = ['handbag', 'backpack']
# CART_CLASS = 'cart'

# cap = cv2.VideoCapture('test_video.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)[0]
#     boxes = results.boxes
#     class_names = results.names

#     dets = []
#     for box in boxes:
#         cls_id = int(box.cls[0])
#         cls_name = class_names[cls_id]
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

#     tracks = tracker.update_tracks(dets, frame=frame)

#     persons = []
#     bags = []
#     items = []

#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         ltrb = track.to_ltrb()
#         cls_name = track.get_det_class()

#         x1, y1, x2, y2 = map(int, ltrb)
#         center = ((x1 + x2) // 2, (y1 + y2) // 2)

#         if cls_name == PERSON_CLASS:
#             persons.append((track_id, (x1, y1, x2, y2), center))
#             color = (255, 255, 0)
#         elif cls_name in BAG_CLASSES:
#             bags.append((track_id, (x1, y1, x2, y2), center))
#             color = (255, 0, 255)
#         elif cls_name != CART_CLASS:
#             items.append((track_id, (x1, y1, x2, y2), center))
#             item_history[track_id].append(center)
#             if len(item_history[track_id]) > 30:
#                 item_history[track_id] = item_history[track_id][-30:]
#             color = (0, 255, 0)
#         else:
#             color = (0, 255, 255)

#         label = f"{cls_name} ID:{track_id}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, label, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # SUSPICIOUS BEHAVIOR LOGIC
#     for item_id, item_box, item_center in items:
#         movement = item_history[item_id]
#         if len(movement) < 10:
#             continue

#         # Check if item moved significantly from its original position (shelf)
#         start_pos = np.array(movement[0])
#         end_pos = np.array(item_center)
#         distance_moved = np.linalg.norm(end_pos - start_pos)
#         if distance_moved < 50:
#             continue  # Not picked up

#         suspicious = False

#         # CASE 1: Placed near pocket area
#         for pid, pbox, pcenter in persons:
#             px1, py1, px2, py2 = pbox
#             pocket_zone_y = py1 + 2 * (py2 - py1) // 3  # lower body
#             if py1 < item_center[1] < py2 and item_center[1] > pocket_zone_y:
#                 suspicious = True
#                 cv2.putText(frame, "⚠️ Suspicious: Into Pocket", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#         # CASE 2: Placed into a bag
#         for _, _, bag_center in bags:
#             bag_dist = np.linalg.norm(np.array(bag_center) - np.array(item_center))
#             if bag_dist < 100:
#                 suspicious = True
#                 cv2.putText(frame, "⚠️ Suspicious: Into Bag", (50, 90),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#     cv2.imshow("Shoplifting Detector", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from collections import defaultdict

# model = YOLO('yolov8m.pt')  # More accurate
# tracker = DeepSort(max_age=60, n_init=2)

# item_history = defaultdict(list)
# track_visibility = defaultdict(int)

# PERSON_CLASS = 'person'
# BAG_CLASSES = ['handbag', 'backpack']
# CART_CLASS = 'cart'

# cap = cv2.VideoCapture('test_video.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)[0]
#     boxes = results.boxes
#     class_names = results.names

#     dets = []
#     for box in boxes:
#         cls_id = int(box.cls[0])
#         cls_name = class_names[cls_id]
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

#     tracks = tracker.update_tracks(dets, frame=frame)

#     persons, bags, items = [], [], []

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         ltrb = track.to_ltrb()
#         cls_name = track.get_det_class()
#         x1, y1, x2, y2 = map(int, ltrb)
#         center = ((x1 + x2) // 2, (y1 + y2) // 2)

#         track_visibility[track_id] += 1

#         if cls_name == PERSON_CLASS:
#             persons.append((track_id, (x1, y1, x2, y2), center))
#             color = (255, 255, 0)
#         elif cls_name in BAG_CLASSES:
#             bags.append((track_id, (x1, y1, x2, y2), center))
#             color = (255, 0, 255)
#         elif cls_name != CART_CLASS:
#             items.append((track_id, (x1, y1, x2, y2), center))
#             item_history[track_id].append(center)
#             if len(item_history[track_id]) > 30:
#                 item_history[track_id] = item_history[track_id][-30:]
#             color = (0, 255, 0)
#         else:
#             color = (0, 255, 255)

#         label = f"{cls_name} ID:{track_id}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     for item_id, item_box, item_center in items:
#         movement = item_history[item_id]
#         if len(movement) < 10:
#             continue

#         start_pos = np.array(movement[0])
#         end_pos = np.array(item_center)
#         distance_moved = np.linalg.norm(end_pos - start_pos)
#         if distance_moved < 50:
#             continue

#         suspicious = False

#         for pid, pbox, pcenter in persons:
#             px1, py1, px2, py2 = pbox
#             pocket_zone_y = py1 + 2 * (py2 - py1) // 3
#             if py1 < item_center[1] < py2 and item_center[1] > pocket_zone_y:
#                 suspicious = True
#                 cv2.putText(frame, "⚠️ Into Pocket", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#         for _, _, bag_center in bags:
#             bag_dist = np.linalg.norm(np.array(bag_center) - np.array(item_center))
#             if bag_dist < 100:
#                 suspicious = True
#                 cv2.putText(frame, "⚠️ Into Bag", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#         if suspicious and track_visibility[item_id] < 10:
#             cv2.putText(frame, "⚠️ Occluded Action", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#     cv2.imshow("Shoplifting Detector", frame)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
