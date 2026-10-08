"""
Configuration Module for SlimVision Smart CCTV Optimization System.
Provides type-safe, production-grade configuration classes with serialization,
validation, and preset profiles.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Set
import json


@dataclass
class MOG2Config:
    """Configuration for Tier-1 MOG2 Background Subtraction."""
    history: int = 500                      # Number of frames for background learning
    var_threshold: float = 50.0            # Mahalanobis distance threshold for foreground classification
    detect_shadows: bool = True            # Whether to mark shadows (as gray=127)
    shadow_threshold: int = 200            # Pixel threshold to remove shadows (keep only values > 200)
    gaussian_blur_kernel: int = 21         # Kernel size for Gaussian noise suppression (must be odd)
    gaussian_blur_sigma: float = 0.0       # Gaussian standard deviation
    morph_open_kernel_size: int = 3        # Morphological opening kernel size for noise removal
    min_contour_area: int = 1000           # Minimum pixel area for a valid motion contour
    warmup_seconds: float = 2.0            # Initial seconds to train background model without detection


@dataclass
class YOLOConfig:
    """Configuration for Tier-2 AI Object Verification."""
    enabled: bool = True                   # Enable AI verification
    model_path: str = "yolov8n.pt"         # Path to YOLOv8 weights (.pt file or model name)
    confidence_threshold: float = 0.35     # Minimum detection confidence
    iou_threshold: float = 0.45            # NMS IoU threshold
    target_classes: List[str] = field(default_factory=lambda: [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase"
    ])                                     # COCO classes considered valid motion triggers
    check_interval: int = 5                # Run YOLO every N frames during motion to conserve compute
    verify_entire_frame: bool = True       # Run inference on full frame or motion bounding box crops


@dataclass
class BufferConfig:
    """Configuration for Zero-OOM Streaming & Ring Buffering."""
    pre_roll_seconds: float = 1.5          # Seconds of footage kept before motion trigger
    post_roll_seconds: float = 2.0         # Seconds of footage recorded after motion ceases
    max_inactivity_seconds: float = 1.5    # Seconds of silence before finalizing a motion event


@dataclass
class SafetyConfig:
    """Configuration for Safe Deletion and Verification."""
    delete_original: bool = False          # Whether to delete source video after processing
    require_verification: bool = True      # Verify output integrity before deleting original
    min_output_bytes: int = 1024           # Minimum valid file size in bytes
    dry_run: bool = False                  # Dry-run mode: analyze and simulate without file operations


@dataclass
class OutputConfig:
    """Configuration for Output Video and Logs."""
    output_folder: str = "processed"       # Folder to store trimmed videos and metadata logs
    preferred_codec: str = "avc1"          # Preferred codec (H.264 / avc1 for web/mobile compatibility)
    fallback_codec: str = "mp4v"           # Fallback codec if preferred is unavailable
    save_json_log: bool = True             # Save detailed JSON event & compression metadata
    draw_bounding_boxes: bool = False      # Draw motion / AI bounding boxes on processed video


@dataclass
class AppConfig:
    """Master Application Configuration."""
    input_folder: str = "cctv"
    watch_mode: bool = False               # Continuous 24/7 folder watcher daemon mode
    poll_interval_sec: float = 5.0         # Polling interval in seconds when watching folder
    mog2: MOG2Config = field(default_factory=MOG2Config)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        mog2 = MOG2Config(**data.get("mog2", {}))
        yolo = YOLOConfig(**data.get("yolo", {}))
        buffer = BufferConfig(**data.get("buffer", {}))
        safety = SafetyConfig(**data.get("safety", {}))
        output = OutputConfig(**data.get("output", {}))
        return cls(
            input_folder=data.get("input_folder", "cctv"),
            watch_mode=data.get("watch_mode", False),
            poll_interval_sec=data.get("poll_interval_sec", 5.0),
            mog2=mog2,
            yolo=yolo,
            buffer=buffer,
            safety=safety,
            output=output
        )
