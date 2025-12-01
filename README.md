# Semi-Autonomous-Vehicle-Perception-Pipeline
This repository contains the complete perception pipeline developed for real-time obstacle detection and monocular distance estimation on low-cost embedded hardware. The system combines YOLOv5 (TensorRT) for detection and MiDaS for dense depth estimation, achieving 8–12 FPS on the NVIDIA Jetson Nano using only an RGB camera.
**Key Features**
*Single-Class Obstacle Detection using YOLOv5s
*Monocular Depth Estimation using MiDaS (MobileNetv3 encoder)
*Distance Computation using Min–Max normalized depth
*TensorRT FP16 Optimized Inference for real-time performance
*Embedded Deployment on NVIDIA Jetson Nano
*Custom Driving Dataset collected at IIT Roorkee

**System Architecture**
<img width="735" height="378" alt="image" src="https://github.com/user-attachments/assets/3d01183f-1800-40bf-975e-cfa6d7fc302a" />

**Dataset**
*Source: Intel RealSense D435 (720p @ 30 FPS)
*Frames: ~13,000
*Annotations: YOLO format (RoboFlow)
*Class: Obstacle (cars, bikes, pedestrians, cones)
*Split: 70% Train / 20% Val / 10% Test

**Models Used**
**YOLOv5 (Object Detection)**
Variant: YOLOv5s
Architecture: CSPDarknet + PANet + YOLO head
Fine-tuned for single-class detection
Converted to ONNX → TensorRT FP16 for high FPS

**MiDaS Small (Depth Estimation)**
Encoder: MobileNetv3
Outputs dense inverse-depth maps
Upsampled to match RGB frame resolution
Depth → Distance conversion:
D = Dmin + (1 - d_rel) * (Dmax - Dmin)
Dmin = 0.5m
Dmax = 20m






