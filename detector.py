"""
Detector Module for SlimVision Smart CCTV Optimization System.
Implements:
1. Tier-1 MotionDetector using OpenCV MOG2 background subtraction.
2. Tier-2 ObjectVerifier using Ultralytics YOLOv8 for AI object confirmation.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from config import MOG2Config, YOLOConfig

logger = logging.getLogger("SlimVision.Detector")


class MotionDetector:
    """
    Tier-1 Fast Motion Detector.
    Uses Gaussian Mixture-based Background/Foreground Segmentation (MOG2).
    """

    def __init__(self, config: MOG2Config):
        self.config = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows
        )
        self.morph_kernel = np.ones(
            (config.morph_open_kernel_size, config.morph_open_kernel_size),
            np.uint8
        )

    def warm_up(self, frame: np.ndarray) -> None:
        """Feed initial frames to allow background model to stabilize."""
        self.bg_subtractor.apply(frame)

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Process a single frame and determine if significant motion exists.

        Returns:
            has_motion (bool): True if contour area exceeds threshold.
            bounding_boxes (List[Tuple[x, y, w, h]]): Motion bounding boxes.
            mask (np.ndarray): Cleaned binary foreground mask.
        """
        # Convert to single-channel grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing to remove high-frequency sensor noise
        k = self.config.gaussian_blur_kernel
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), self.config.gaussian_blur_sigma)

        # Apply MOG2 background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)

        # Threshold to eliminate shadows (shadows are marked as 127 by MOG2)
        _, thresh_mask = cv2.threshold(
            fg_mask,
            self.config.shadow_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Morphological opening (erosion followed by dilation) to remove speckle noise
        clean_mask = cv2.morphologyEx(
            thresh_mask,
            cv2.MORPH_OPEN,
            self.morph_kernel
        )

        # Find external contours of moving regions
        contours, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_boxes = []
        has_motion = False

        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.config.min_contour_area:
                has_motion = True
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append((x, y, w, h))

        return has_motion, motion_boxes, clean_mask


class ObjectVerifier:
    """
    Tier-2 AI Object Verifier.
    Uses Ultralytics YOLOv8 to verify whether detected motion contains actual objects of interest
    (e.g., persons, vehicles, animals) and eliminate environmental false positives.
    """

    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.available = False
        self.target_classes_set = set(c.lower() for c in config.target_classes)

        if not config.enabled:
            logger.info("YOLOv8 Object Verification is disabled by configuration.")
            return

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load YOLOv8 model weights."""
        try:
            from ultralytics import YOLO
            import os

            model_path = self.config.model_path
            # Check if local model file exists, else use YOLO default weights name
            if not os.path.exists(model_path) and os.path.exists("yolov8n.pt"):
                model_path = "yolov8n.pt"

            logger.info(f"Loading YOLOv8 model from: {model_path}...")
            self.model = YOLO(model_path)
            self.available = True
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not initialize YOLOv8 model ({e}). "
                "Falling back to pure MOG2 motion detection."
            )
            self.available = False

    def verify(self, frame: np.ndarray, motion_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify presence of target objects in the frame or motion regions.

        Returns:
            is_verified (bool): True if at least one target class is detected with sufficient confidence.
            detections (List[Dict]): List of detected objects with label, confidence, and box (x1, y1, x2, y2).
        """
        if not self.available or self.model is None:
            # If YOLO is disabled or unavailable, default to approving motion detection
            return True, []

        try:
            # Run inference on frame
            results = self.model.predict(
                source=frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False
            )

            verified_detections = []
            is_verified = False

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = result.names.get(cls_id, str(cls_id)).lower()
                        conf = float(box.conf[0].item())

                        if len(self.target_classes_set) == 0 or cls_name in self.target_classes_set:
                            is_verified = True
                            xyxy = box.xyxy[0].tolist()
                            verified_detections.append({
                                "class": cls_name,
                                "confidence": round(conf, 3),
                                "box": [round(coord, 1) for coord in xyxy]
                            })

            return is_verified, verified_detections

        except Exception as e:
            logger.error(f"Error during YOLOv8 inference: {e}. Defaulting to motion approval.")
            return True, []

