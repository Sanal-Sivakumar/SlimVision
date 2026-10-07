# SlimVision: AI-Powered Smart CCTV Video & Storage Optimization System

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-MOG2-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Executive Summary & Problem Statement

Standard Closed-Circuit Television (CCTV) systems record 24/7 continuous video streams. In most enterprise, commercial, and residential surveillance environments, **over 80% to 95% of recorded footage consists of static, empty scenes** with zero actionable activity.

### The Cost of Redundant Footage:
- **Excessive Storage Demands**: Exabytes of static footage fill on-premise NVR hard drives and inflate cloud storage bills.
- **Hardware Wear & Power Consumption**: Continuous, high-bitrate write cycles accelerate drive failure rates and increase server power consumption.
- **Investigation Inefficiencies**: Security analysts must spend hours manually scrubbing through hours of empty recordings to find key events.

### The SlimVision Solution:
SlimVision is an industrial-grade, two-tier video optimization pipeline. It combines **high-speed background subtraction (MOG2)** with **deep learning object verification (YOLOv8)** and **zero-OOM streaming ring buffers**. It automatically filters out empty frames, preserves smooth pre-event and post-event context, logs granular forensic metadata, and safely reduces video storage footprints by up to 90%.

---

## 2. System Architecture

SlimVision utilizes a hierarchical two-tier detection architecture designed for high throughput and low false-positive rates:

```mermaid
flowchart TD
    A["Raw CCTV Video (MP4/AVI/MKV)"] --> B["cv2.VideoCapture Stream"]
    B --> C["Warmup Phase (First 2.0s)"]
    C --> D["Grayscale & Gaussian Noise Filter"]
    D --> E["Tier-1: MOG2 Background Subtraction"]
    E --> F["Shadow Stripping & Morphological Opening"]
    F --> G{"Contour Area > Threshold?"}
    G -- "No (Static Frame)" --> H["Buffer in Pre-Roll Ring Buffer (1.5s)"]
    G -- "Yes (Motion Detected)" --> I["Tier-2: YOLOv8 AI Verification"]
    I --> J{"Target Class Verified? (Person, Car, etc.)"}
    J -- "No (False Alarm: Wind/Shadows)" --> H
    J -- "Yes (Confirmed Event)" --> K["Flush Pre-Roll Buffer to Stream"]
    K --> L["Write Active Frame Directly to Disk"]
    L --> M["Post-Roll Cooldown (2.0s)"]
    M --> N["Generate JSON Metadata & Audit Log"]
    N --> O{"Integrity Verification Passed?"}
    O -- "Yes & --delete-original" --> P["Safely Unlink Original Video"]
    O -- "No" --> Q["Preserve Source Footage & Alert"]
```

---

## 3. Key Features

- **Hierarchical Two-Tier Detection**:
  - *Tier 1 (MOG2)*: Ultra-fast $O(1)$ pixel-level background subtraction discards static frames without GPU overhead.
  - *Tier 2 (YOLOv8)*: AI verification eliminates false triggers caused by swaying branches, moving clouds, shadows, and headlights.
- **Zero-OOM Streaming Ring Buffer**:
  - Eliminates in-memory frame accumulation crashes.
  - Preserves **Pre-Roll** (1.5s prior to trigger) and **Post-Roll** (2.0s after motion ceases) to ensure context is never clipped.
- **Industrial Safe Deletion Protocol**:
  - Never deletes raw footage blindly.
  - Verifies container validity, frame count, byte size, and JSON log integrity before safely removing source files.
- **Universal Codec Negotiation**:
  - Automatically selects `avc1` (H.264) for web/mobile streaming playback, falling back to `mp4v` if needed.
- **Batch & Directory Processing**:
  - Recursively scans input directories, processes videos sequentially, and outputs consolidated batch metrics.

---

## 4. Installation & Setup

### Prerequisites
- Linux / macOS / Windows
- Python 3.8 or higher
- FFmpeg (optional, recommended for hardware acceleration)

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd SlimVision

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# Or manually:
pip install opencv-python-headless ultralytics torch torchvision tqdm
```

---

## 5. Usage & CLI Command Reference

### Basic Run (Batch Directory Processing)
Process all videos in `cctv/` and output optimized videos to `processed/`:
```bash
python main.py
```

### Process Single Video File
```bash
python main.py -i /path/to/security_cam_01.mp4 -o /path/to/output_folder
```

### Dry Run (Simulate Savings without Writing or Deleting)
```bash
python main.py -i cctv/ --dry-run
```

### Enable Verified Safe Deletion
```bash
python main.py -i cctv/ --delete-original
```

### Custom AI Verification & Filtering
```bash
# Only record when persons, cars, or dogs are verified
python main.py -i cctv/ --confidence 0.40 --classes person,car,truck,dog

# Run pure motion mode without YOLO (lightweight CPU mode)
python main.py -i cctv/ --no-yolo
```

### Command-Line Arguments Table

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | `cctv` | Path to video file or directory containing video files. |
| `-o`, `--output` | `processed` | Directory where optimized videos and logs will be saved. |
| `--model` | `yolov8n.pt` | Path or name of YOLOv8 weights (`yolov8n.pt`, `yolov8s.pt`, etc.). |
| `--no-yolo` | `False` | Disable YOLOv8 and operate in fast MOG2 motion mode. |
| `--confidence` | `0.35` | Minimum detection confidence threshold for YOLOv8. |
| `--classes` | `person,car...` | Comma-separated list of target object classes to record. |
| `--min-area` | `1000` | Minimum contour area (pixels) to qualify as Tier-1 motion. |
| `--pre-roll` | `1.5` | Seconds of footage captured before motion trigger. |
| `--post-roll` | `2.0` | Seconds of footage captured after motion ceases. |
| `--delete-original` | `False` | Safely remove original video after integrity verification. |
| `--dry-run` | `False` | Analyze footage and output metrics without file deletion. |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 6. Output Format & JSON Metadata Schema

Each processed video produces:
1. **Optimized Video**: `<filename>_optimized.mp4` (containing only verified motion events with pre/post-roll).
2. **Metadata Log**: `<filename>_log.json` containing complete forensic timestamps and compression analytics.

### Sample JSON Output (`output_log.json`):
```json
{
    "source_file": "/workspace/cctv/cam01.mp4",
    "output_video": "/workspace/processed/cam01_optimized.mp4",
    "codec_used": "avc1",
    "metrics": {
        "original_frames": 9000,
        "compressed_frames": 1800,
        "original_duration_sec": 300.0,
        "compressed_duration_sec": 60.0,
        "temporal_reduction_percentage": 80.0,
        "original_file_size_mb": 75.4,
        "compressed_file_size_mb": 15.1,
        "storage_reduction_percentage": 79.97,
        "total_events_detected": 3,
        "processing_time_sec": 14.2,
        "processing_fps": 211.2,
        "realtime_factor": 7.04
    },
    "events": [
        {
            "event_id": 1,
            "start_seconds": 12.4,
            "end_seconds": 32.1,
            "duration_seconds": 19.7,
            "start_frame": 372,
            "end_frame": 963,
            "detected_objects": [
                {
                    "class": "person",
                    "max_confidence": 0.892
                },
                {
                    "class": "car",
                    "max_confidence": 0.764
                }
            ]
        }
    ]
}
```

---

## 7. Documentation Roadmap

For in-depth explanations, refer to the documentation files:
- 📖 **[`technical_details.md`](file:///home/sanal-sivakumar/Documents/smart-cctv/technical_details.md)**: Deep pedagogical breakdown of every algorithm, math formula, computer vision concept, and architectural design decision.
- 🛠️ **[`troubleshoot.md`](file:///home/sanal-sivakumar/Documents/smart-cctv/troubleshoot.md)**: Production error handbook covering common bugs, codec failures, edge-case mitigation, and performance tuning.

---

## 8. Authors & Contributors

- **Amrutha M**
- **Sanal Sivakumar**
- **Megha Suresh**
- **Agnivesh S**
