"""
Pipeline Module for SlimVision Smart CCTV Optimization System.
Implements the core video processing pipeline featuring:
- Streaming video writer with zero-OOM ring buffering (pre-roll & post-roll)
- Two-tier motion filtering & AI object verification
- Robust codec negotiation (avc1/mp4v)
- Detailed compression metrics & JSON logging
- Industrial-grade safe deletion with video integrity validation
"""

import os
import time
import logging
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import AppConfig
from detector import MotionDetector, ObjectVerifier

logger = logging.getLogger("SlimVision.Pipeline")


class VideoOptimizerPipeline:
    """
    Industrial-grade video optimization pipeline for CCTV footage.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.motion_detector = MotionDetector(config.mog2)
        self.object_verifier = ObjectVerifier(config.yolo)

    def _get_video_writer(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int
    ) -> Tuple[Optional[cv2.VideoWriter], str]:
        """
        Create a VideoWriter with codec negotiation (prefers avc1/H.264, falls back to mp4v).
        """
        codecs_to_try = [
            (self.config.output.preferred_codec, f"*{self.config.output.preferred_codec}"),
            (self.config.output.fallback_codec, f"*{self.config.output.fallback_codec}"),
            ("mp4v", "*mp4v"),
            ("XVID", "*XVID")
        ]

        # Deduplicate while preserving order
        seen = set()
        unique_codecs = []
        for name, fourcc_str in codecs_to_try:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_codecs.append((name, fourcc_str))

        for name, fourcc_str in unique_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*name)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    logger.debug(f"Initialized VideoWriter using codec '{name}' at {output_path}")
                    return writer, name
            except Exception as e:
                logger.debug(f"Codec '{name}' initialization failed: {e}")

        logger.error(f"Failed to initialize VideoWriter for {output_path} with any available codec.")
        return None, "none"

    def process_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single CCTV video file through the optimization pipeline.

        Returns:
            summary (Dict): Complete metadata, metrics, and event log.
        """
        start_time = time.time()
        logger.info(f"Starting processing: {video_path}")

        if not os.path.exists(video_path):
            logger.error(f"Input video does not exist: {video_path}")
            return None

        # Gather source file metadata
        original_file_size = os.path.getsize(video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Could not open video stream: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            logger.warning(f"Invalid FPS detected ({fps}). Falling back to 30.0.")
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_duration = total_frames / fps if total_frames > 0 else 0.0

        logger.info(
            f"Video properties: {width}x{height} @ {fps:.2f} FPS | "
            f"Total Frames: {total_frames} | Duration: {original_duration:.2f}s | "
            f"Size: {original_file_size / (1024 * 1024):.2f} MB"
        )

        # Prepare output paths
        filename = os.path.basename(video_path)
        base_name, _ = os.path.splitext(filename)
        os.makedirs(self.config.output.output_folder, exist_ok=True)
        output_video_path = os.path.join(
            self.config.output.output_folder,
            f"{base_name}_optimized.mp4"
        )
        output_log_path = os.path.join(
            self.config.output.output_folder,
            f"{base_name}_log.json"
        )

        # Buffering parameters
        pre_roll_limit = max(1, int(fps * self.config.buffer.pre_roll_seconds))
        post_roll_limit = max(1, int(fps * self.config.buffer.post_roll_seconds))
        warmup_frames = int(fps * self.config.mog2.warmup_seconds)

        pre_roll_buffer = deque(maxlen=pre_roll_limit)
        events: List[Dict[str, Any]] = []

        writer: Optional[cv2.VideoWriter] = None
        used_codec = "none"
        frames_written = 0

        # State tracking
        is_recording = False
        event_start_frame = 0
        event_start_sec = 0.0
        post_roll_counter = 0
        current_event_objects: Dict[str, float] = {}

        pbar = tqdm(total=total_frames, desc=f"Processing {filename}", unit="frame")
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                pbar.update(1)

                # Warmup phase: train background subtractor without saving
                if frame_idx <= warmup_frames:
                    self.motion_detector.warm_up(frame)
                    continue

                current_sec = frame_idx / fps

                # Tier-1 Motion Detection
                has_motion, motion_boxes, _ = self.motion_detector.detect(frame)
                motion_verified = False
                detected_objects = []

                if has_motion:
                    # Tier-2 AI Object Verification (if enabled)
                    if self.config.yolo.enabled and self.object_verifier.available:
                        # Check periodically during active motion or on initial trigger
                        if not is_recording or (frame_idx % self.config.yolo.check_interval == 0):
                            motion_verified, detected_objects = self.object_verifier.verify(
                                frame,
                                motion_boxes if not self.config.yolo.verify_entire_frame else None
                            )
                        else:
                            # Keep active during existing verified motion sequence
                            motion_verified = is_recording
                    else:
                        motion_verified = True

                # Record any detected objects
                for obj in detected_objects:
                    cls_name = obj["class"]
                    conf = obj["confidence"]
                    current_event_objects[cls_name] = max(
                        current_event_objects.get(cls_name, 0.0),
                        conf
                    )

                # ==========================================
                # Finite State Machine for Video Recording
                # ==========================================
                if motion_verified:
                    if not is_recording:
                        # Transition: IDLE -> RECORDING
                        is_recording = True
                        event_start_frame = max(1, frame_idx - len(pre_roll_buffer))
                        event_start_sec = event_start_frame / fps
                        post_roll_counter = 0

                        # Lazy initialize VideoWriter on first verified motion event
                        if writer is None:
                            writer, used_codec = self._get_video_writer(
                                output_video_path, fps, width, height
                            )

                        # Flush pre-roll circular buffer to ensure smooth pre-event context
                        if writer is not None:
                            while pre_roll_buffer:
                                buffered_frame = pre_roll_buffer.popleft()
                                writer.write(buffered_frame)
                                frames_written += 1

                    # Write current active frame
                    if writer is not None:
                        writer.write(frame)
                        frames_written += 1
                    post_roll_counter = 0

                else:
                    if is_recording:
                        # In cooldown / post-roll phase
                        post_roll_counter += 1
                        if writer is not None:
                            writer.write(frame)
                            frames_written += 1

                        if post_roll_counter >= post_roll_limit:
                            # Transition: RECORDING -> IDLE (Event Complete)
                            event_end_frame = frame_idx
                            event_end_sec = event_end_frame / fps
                            duration = event_end_sec - event_start_sec

                            events.append({
                                "event_id": len(events) + 1,
                                "start_seconds": round(event_start_sec, 3),
                                "end_seconds": round(event_end_sec, 3),
                                "duration_seconds": round(duration, 3),
                                "start_frame": event_start_frame,
                                "end_frame": event_end_frame,
                                "detected_objects": [
                                    {"class": k, "max_confidence": round(v, 3)}
                                    for k, v in current_event_objects.items()
                                ]
                            })

                            is_recording = False
                            post_roll_counter = 0
                            current_event_objects = {}
                            pre_roll_buffer.clear()
                    else:
                        # Append to circular pre-roll buffer in idle state
                        pre_roll_buffer.append(frame)

            # ==========================================
            # End-of-Stream Handling (Flush active event)
            # ==========================================
            if is_recording:
                event_end_frame = frame_idx
                event_end_sec = event_end_frame / fps
                duration = event_end_sec - event_start_sec

                events.append({
                    "event_id": len(events) + 1,
                    "start_seconds": round(event_start_sec, 3),
                    "end_seconds": round(event_end_sec, 3),
                    "duration_seconds": round(duration, 3),
                    "start_frame": event_start_frame,
                    "end_frame": event_end_frame,
                    "detected_objects": [
                        {"class": k, "max_confidence": round(v, 3)}
                        for k, v in current_event_objects.items()
                    ]
                })

        finally:
            pbar.close()
            cap.release()
            if writer is not None:
                writer.release()

        elapsed_time = time.time() - start_time
        processed_file_size = os.path.getsize(output_video_path) if os.path.exists(output_video_path) else 0
        compressed_duration = frames_written / fps if frames_written > 0 else 0.0

        # Compute performance and savings metrics
        temporal_reduction_pct = (
            ((original_duration - compressed_duration) / original_duration * 100.0)
            if original_duration > 0 else 0.0
        )
        storage_reduction_pct = (
            ((original_file_size - processed_file_size) / original_file_size * 100.0)
            if original_file_size > 0 and processed_file_size > 0 else 0.0
        )
        processing_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0.0

        summary = {
            "source_file": os.path.abspath(video_path),
            "output_video": os.path.abspath(output_video_path) if processed_file_size > 0 else None,
            "codec_used": used_codec,
            "metrics": {
                "original_frames": frame_idx,
                "compressed_frames": frames_written,
                "original_duration_sec": round(original_duration, 2),
                "compressed_duration_sec": round(compressed_duration, 2),
                "temporal_reduction_percentage": round(temporal_reduction_pct, 2),
                "original_file_size_mb": round(original_file_size / (1024 * 1024), 2),
                "compressed_file_size_mb": round(processed_file_size / (1024 * 1024), 2),
                "storage_reduction_percentage": round(storage_reduction_pct, 2),
                "total_events_detected": len(events),
                "processing_time_sec": round(elapsed_time, 2),
                "processing_fps": round(processing_fps, 2),
                "realtime_factor": round(processing_fps / fps, 2) if fps > 0 else 1.0
            },
            "events": events
        }

        # Save metadata log
        if self.config.output.save_json_log:
            with open(output_log_path, "w") as f:
                import json
                json.dump(summary, f, indent=4)
            logger.info(f"Saved metadata log to {output_log_path}")

        # Safe Deletion Validation
        self._verify_and_safe_delete(video_path, output_video_path, output_log_path, frames_written)

        return summary

    def _verify_and_safe_delete(
        self,
        input_path: str,
        output_path: str,
        log_path: str,
        frames_written: int
    ) -> bool:
        """
        Industrial-grade safe deletion protocol.
        Ensures output video and log pass strict integrity checks before deleting the original.
        """
        if not self.config.safety.delete_original:
            logger.info("Original file preserved (delete_original is False).")
            return False

        if self.config.safety.dry_run:
            logger.info(f"[DRY-RUN] Would have safely deleted original: {input_path}")
            return False

        logger.info(f"Executing Safe Deletion Protocol for: {input_path}")

        # 1. Check if output video exists and exceeds minimum size
        if frames_written > 0:
            if not os.path.exists(output_path):
                logger.error(f"Safe-delete aborted: Output file {output_path} does not exist.")
                return False

            out_size = os.path.getsize(output_path)
            if out_size < self.config.safety.min_output_bytes:
                logger.error(
                    f"Safe-delete aborted: Output file {output_path} size ({out_size} bytes) "
                    f"below minimum threshold ({self.config.safety.min_output_bytes} bytes)."
                )
                return False

            # 2. Re-open output video to verify container validity
            if self.config.safety.require_verification:
                verify_cap = cv2.VideoCapture(output_path)
                if not verify_cap.isOpened():
                    logger.error(f"Safe-delete aborted: Output video {output_path} is corrupt or unreadable.")
                    return False
                v_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                verify_cap.release()

                if v_frames <= 0:
                    logger.error(f"Safe-delete aborted: Verified frame count in {output_path} is {v_frames}.")
                    return False

        # 3. Verify JSON log file exists
        if not os.path.exists(log_path):
            logger.error(f"Safe-delete aborted: Log file {log_path} not found.")
            return False

        # 4. Perform deletion
        try:
            os.remove(input_path)
            logger.info(f"✅ Original file safely removed: {input_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete original file {input_path}: {e}")
            return False

