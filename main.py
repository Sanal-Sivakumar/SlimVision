"""
Main CLI Entry Point for SlimVision Smart CCTV Optimization System.

Features:
- Industrial CLI argument parsing
- Directory batch processing or single video processing
- 24x7 Continuous Folder Watcher Daemon Mode (--watch / -w)
- Signal handling for graceful termination (SIGINT / SIGTERM)
- Configuration override via CLI
"""

import os
import sys
import time
import argparse
import signal
import logging
from typing import List, Set

from config import AppConfig, MOG2Config, YOLOConfig, BufferConfig, SafetyConfig, OutputConfig
from pipeline import VideoOptimizerPipeline

# Global shutdown flag
RUNNING = True


def setup_logging(log_level: str = "INFO"):
    """Configure enterprise-grade structured logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = [handler]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SlimVision: Industrial Smart CCTV Motion & AI Video Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="cctv",
        help="Path to input video file or folder containing CCTV videos"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="processed",
        help="Path to output folder for processed videos and JSON logs"
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Enable continuous 24/7 folder watcher daemon mode"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds when running in 24/7 watch mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 weights file or model name (e.g., yolov8n.pt, yolov8s.pt)"
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLOv8 AI object verification and use pure MOG2 motion detection (fastest CPU mode)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLOv8 confidence threshold for object detection"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,bird,cat,dog,horse,sheep,cow,backpack,handbag,suitcase",
        help="Comma-separated list of target classes to verify"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum contour pixel area for Tier-1 MOG2 motion filtering"
    )
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=1.5,
        help="Pre-roll buffer duration in seconds (recorded prior to motion trigger)"
    )
    parser.add_argument(
        "--post-roll",
        type=float,
        default=2.0,
        help="Post-roll buffer duration in seconds (recorded after motion ceases)"
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Enable verified safe-deletion of original video files after processing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate processing and statistics without modifying or deleting input files"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def get_video_files(input_path: str) -> List[str]:
    """Discover video files from input path."""
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if os.path.isfile(input_path):
        _, ext = os.path.splitext(input_path)
        if ext.lower() in valid_extensions:
            return [input_path]
        return []

    if os.path.isdir(input_path):
        files = []
        for root, _, filenames in os.walk(input_path):
            for f in sorted(filenames):
                _, ext = os.path.splitext(f)
                if ext.lower() in valid_extensions and not f.startswith("."):
                    files.append(os.path.join(root, f))
        return files

    return []


def run_batch(pipeline: VideoOptimizerPipeline, videos: List[str], logger: logging.Logger) -> None:
    """Execute optimization on a list of videos."""
    total_orig_mb = 0.0
    total_comp_mb = 0.0
    total_events = 0

    for idx, video in enumerate(videos, 1):
        if not RUNNING:
            logger.info("Processing interrupted. Exiting batch queue.")
            break

        logger.info(f"\n--- [{idx}/{len(videos)}] Processing: {video} ---")
        try:
            summary = pipeline.process_video(video)
            if summary:
                m = summary["metrics"]
                total_orig_mb += m["original_file_size_mb"]
                total_comp_mb += m["compressed_file_size_mb"]
                total_events += m["total_events_detected"]
                logger.info(
                    f"✓ Completed: {video} | "
                    f"Temporal Reduction: {m['temporal_reduction_percentage']}% | "
                    f"Storage Savings: {m['storage_reduction_percentage']}% | "
                    f"Events: {m['total_events_detected']}"
                )
        except Exception as e:
            logger.error(f"Error processing video {video}: {e}", exc_info=True)

    overall_savings_pct = (
        ((total_orig_mb - total_comp_mb) / total_orig_mb * 100.0)
        if total_orig_mb > 0 else 0.0
    )
    logger.info("\n=======================================================")
    logger.info("  BATCH PROCESSING SUMMARY                              ")
    logger.info("=======================================================")
    logger.info(f"Videos Processed:    {len(videos)}")
    logger.info(f"Total Motion Events: {total_events}")
    logger.info(f"Original Size:       {total_orig_mb:.2f} MB")
    logger.info(f"Optimized Size:      {total_comp_mb:.2f} MB")
    logger.info(f"Storage Saved:       {(total_orig_mb - total_comp_mb):.2f} MB ({overall_savings_pct:.2f}%)")
    logger.info("=======================================================\n")


def main():
    global RUNNING
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("SlimVision.Main")

    logger.info("=======================================================")
    logger.info("  SlimVision: Smart CCTV Motion & AI Video Optimizer   ")
    logger.info("=======================================================")

    def signal_handler(sig, frame):
        global RUNNING
        logger.warning("\nShutdown signal received. Finishing current task and exiting gracefully...")
        RUNNING = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Build typed configuration
    target_classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    config = AppConfig(
        input_folder=args.input,
        watch_mode=args.watch,
        poll_interval_sec=args.poll_interval,
        mog2=MOG2Config(
            min_contour_area=args.min_area
        ),
        yolo=YOLOConfig(
            enabled=not args.no_yolo,
            model_path=args.model,
            confidence_threshold=args.confidence,
            target_classes=target_classes
        ),
        buffer=BufferConfig(
            pre_roll_seconds=args.pre_roll,
            post_roll_seconds=args.post_roll
        ),
        safety=SafetyConfig(
            delete_original=args.delete_original,
            dry_run=args.dry_run
        ),
        output=OutputConfig(
            output_folder=args.output
        )
    )

    pipeline = VideoOptimizerPipeline(config)

    # Ensure input directory exists
    os.makedirs(args.input, exist_ok=True)

    if args.watch:
        logger.info(f"Starting 24/7 Watcher Daemon on folder: '{args.input}'")
        logger.info(f"Polling interval: {args.poll_interval}s. Press Ctrl+C or send SIGTERM to stop.")

        processed_set: Set[str] = set()

        while RUNNING:
            videos = get_video_files(args.input)
            # Filter for unhandled videos
            new_videos = [v for v in videos if v not in processed_set]

            if new_videos:
                logger.info(f"Detected {len(new_videos)} new video(s) to optimize.")
                run_batch(pipeline, new_videos, logger)
                for v in new_videos:
                    if not config.safety.delete_original:
                        processed_set.add(v)

            # Sleep with interrupt responsiveness
            for _ in range(int(args.poll_interval * 10)):
                if not RUNNING:
                    break
                time.sleep(0.1)

        logger.info("24/7 Watcher Daemon stopped safely.")

    else:
        # Standard Single/Batch Run
        videos = get_video_files(args.input)
        if not videos:
            logger.warning(f"No video files found at input location: '{args.input}'")
            logger.info(f"Please place video files in '{args.input}' or run with -w to watch for new files.")
            return

        logger.info(f"Found {len(videos)} video file(s) to process.")
        run_batch(pipeline, videos, logger)


if __name__ == "__main__":
    main()