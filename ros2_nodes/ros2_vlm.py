#!/usr/bin/env python3
import os
import time
import threading
import queue

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from neo4j.exceptions import CypherSyntaxError

from BIM_processing.KG_to_VLM import VLMGraphRAGPipeline
import base64

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node has been started!')

        # Subscriptions (Isaac Sim default)
        self.create_subscription(Image, "/rgb", self.image_callback, 10)
        # Gazebo example:
        # self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        self.bridge = CvBridge()
        self.llm_model = "llama3.2-vision:11b"

        #neo4j_auth = ("neo4j", "kg_bim_cranehall") , neo4j_auth=("neo4j", "riedelbau_model")

        self.vlm_cli = VLMGraphRAGPipeline(
            neo4j_uri="neo4j://127.0.0.1:7687",
            neo4j_auth = ("neo4j", "CRANE_HALL"),
            llm_model=self.llm_model
        )

        # Keep only the latest frame (bytes)
        self.vlm_queue = queue.Queue(maxsize=1)

        # Background worker thread for VLM processing
        self.processing_thread = threading.Thread(target=self.vlm_worker, daemon=True)
        self.processing_thread.start()

    @staticmethod
    def bgr_to_rgb(cv_image: np.ndarray) -> np.ndarray:
        """Convert an OpenCV image from BGR to RGB format."""
        if cv_image is None or not hasattr(cv_image, "shape"):
            raise ValueError("Invalid OpenCV image (None or missing shape).")
        if cv_image.size == 0:
            raise ValueError("Empty OpenCV image received.")
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def cv_image_to_jpeg_bytes(cv_image: np.ndarray, quality: int = 90, max_side: int = 1280) -> bytes:
        """
        Convert a cv2 image (RGB NumPy array is fine) into JPEG bytes (in-memory, no files).
        """
        if cv_image is None or not hasattr(cv_image, "shape") or cv_image.size == 0:
            raise ValueError("Invalid or empty image")

        if cv_image.dtype != np.uint8:
            cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)

        h, w = cv_image.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            cv_image = cv2.resize(cv_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        b64_str = base64.b64encode(buf).decode('utf-8')
        
        if not ok:
            raise RuntimeError("cv2.imencode(.jpg) failed")

        return b64_str

    def image_callback(self, msg: Image):
        """
        Fast callback: ROS Image -> OpenCV (BGR) -> RGB -> JPEG bytes, queue latest.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = self.bgr_to_rgb(cv_image)
            base64_str = self.cv_image_to_jpeg_bytes(cv_image)


            # Drop stale frame if worker hasn't consumed it yet
            if not self.vlm_queue.empty():
                try:
                    _ = self.vlm_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.vlm_queue.put_nowait(base64_str)
            except queue.Full:
                self.get_logger().warn('Queue full, frame dropped')

        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")
            import traceback; traceback.print_exc()

    def vlm_worker(self):
        """
        Background loop: waits for JPEG bytes and calls the VLM pipeline with image_bytes.
        """
        self.get_logger().info('VLM worker thread started.')

        while rclpy.ok():
            try:
                base64_str = self.vlm_queue.get(timeout=1.0)  # <-- bytes, not ndarray
                self.get_logger().info(f'Processing frame ({len(base64_str)} bytes JPEG) with VLM...')

                user_question = (
                    "can you list the doors in the building with near positions to this positions (-7.563,0.15388,-0.4889), use euclidean distance to calculate the distance" 
                )

                # IMPORTANT: pass bytes via keyword the pipeline supports
                self.vlm_cli.run(self.llm_model, user_question)

                self.get_logger().info('VLM pipeline completed.')

            except CypherSyntaxError as e:
                self.get_logger().error(f"Cypher syntax error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in VLM worker: {e}")
                import traceback; traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
