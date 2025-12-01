#!/usr/bin/env python3
# YOLO + MiDaS live inference using Intel RealSense L515 (Jetson Nano compatible)

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt
from yoloDet import YoloTRT

# -------------------------------
# 1. Load MiDaS model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

print("[INFO] Loading MiDaS Small model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# -------------------------------
# 2. Load YOLO TensorRT engine
# -------------------------------
yolo_engine = "yolov5/build/yolov5s.engine"
yolo_lib = "yolov5/build/libmyplugins.so"
yolo = YoloTRT(library=yolo_lib, engine=yolo_engine, conf=0.5, yolo_ver="v5")
print("[INFO] YOLOv5 TensorRT engine loaded successfully.")

# -------------------------------
# 3. Setup Intel RealSense pipeline
# -------------------------------
pipeline = rs.pipeline()
config = rs.config()

# Valid L515 configuration
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
try:
    pipeline.start(config)
    print("[INFO] RealSense L515 streaming started successfully.")
except Exception as e:
    print("[ERROR] Failed to start RealSense pipeline:", e)
    exit(1)

# -------------------------------
# 4. Helper functions
# -------------------------------
def scale_depth_to_meters(depth_rel, D_min=2.0, D_max=15.0):
    """
    Convert MiDaS relative depth (larger = farther) to approximate metric distances.
    Red = close, blue = far, with more weight given to farther (bluer) regions.
    """
    depth_norm = (depth_rel - np.min(depth_rel)) / (np.max(depth_rel) - np.min(depth_rel) + 1e-6)
    # Inverse scaling so smaller MiDaS value → smaller distance
    depth_m = D_max - np.power(depth_norm, 1.3) * (D_max - D_min)
    return depth_m


def get_box_distance(box, depth_map):
    h, w = depth_map.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return float(np.mean(crop))


def draw_distance(frame, box, dist_m, color=(0, 255, 0)):
    """Draw bounding box and display distance at the bottom-left corner."""
    if dist_m is None:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    label = f"{dist_m:.1f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    tx, ty = x1 + 4, y2 - 4
    bg_pt1 = (tx - 2, ty - th - baseline - 2)
    bg_pt2 = (tx + tw + 2, ty + baseline + 2)
    cv2.putText(frame, label, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)


# -------------------------------
# 5. Main loop
# -------------------------------
print("[INFO] Press 'q' to exit.")
cmap = plt.get_cmap('jet')
prev_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- MiDaS depth estimation ---
        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze(0).squeeze(0)
            output = prediction.cpu().numpy()

        # Scale depth
        depth_m = scale_depth_to_meters(output)
        depth_resized = cv2.resize(depth_m, (color_image.shape[1], color_image.shape[0]))

        # --- Visualize depth ---
        depth_clipped = np.clip(depth_resized, 2.0, 15.0)
        depth_norm = ((depth_clipped - 2.0) / (15.0 - 2.0) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # --- Blend RGB + depth ---
        combo = cv2.addWeighted(color_image, 0.7, depth_colormap, 0.3, 0)

        # --- YOLO detection + distance ---
        detections, _ = yolo.Inference(color_image)
        for obj in detections:
            box = obj.get("box")
            class_id = obj.get("class_id", 0)
            conf = obj.get("conf", 0)
            label = f"ID:{class_id} {conf:.2f}"

    # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(combo, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(combo, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
            dist_m = get_box_distance(box, depth_resized)
            draw_distance(combo, box, dist_m)

        # FPS counter
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(combo, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("YOLO + MiDaS + RealSense L515", combo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Stream stopped successfully.")
