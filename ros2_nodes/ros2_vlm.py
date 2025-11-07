#!/usr/bin/env python3
import os
import time
import base64
import threading
import queue

import rclpy
from rclpy.node import Node

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from neo4j.exceptions import CypherSyntaxError

# keep your import as-is
from BIM_processing.KG_to_VLM import VLMGraphRAGPipeline


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

        # DO NOT change your Neo4j settings (per your request)
        self.vlm_cli = VLMGraphRAGPipeline(
            neo4j_uri="neo4j://127.0.0.1:7687",
            neo4j_auth=("neo4j", "riedelbau_model"),
            llm_model=self.llm_model
        )

        # Keep only the most recent frame to avoid backlog
        self.vlm_queue = queue.Queue(maxsize=1)

        # Single background worker thread for VLM processing
        self.processing_thread = threading.Thread(target=self.vlm_worker, daemon=True)
        self.processing_thread.start()

    @staticmethod
    def _to_b64_jpeg(cv_image, max_side=1280, quality=90) -> str:
        """
        Convert an OpenCV BGR image to a downscaled JPEG base64 string.
        Downscaling reduces latency and context size for VLMs.
        """
        h, w = cv_image.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            cv_image = cv2.resize(cv_image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf).decode("utf-8")

    def image_callback(self, msg: Image):
        """
        Fast callback: convert ROS Image -> OpenCV -> base64 JPEG and queue it.
        Keeps only the newest frame.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            b64_jpeg = self._to_b64_jpeg(cv_image)

            # Drop stale frame if worker hasn't consumed it yet
            if not self.vlm_queue.empty():
                try:
                    self.vlm_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.vlm_queue.put_nowait(b64_jpeg)
            except queue.Full:
                # Should be rare since we just popped; safe-guard anyway
                pass

        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")

    def vlm_worker(self):
        """
        Background loop: waits for a base64 image and queries the VLM pipeline.
        """
        self.get_logger().info('VLM worker thread started.')
        while rclpy.ok():
            try:
                # Block briefly for new frame
                b64_image = self.vlm_queue.get(timeout=1.0)

                user_question = (
                    "can you fetch the doors from the data provided and then take a look at the image and tell me if you see a door or not? "
                    "if you don't know please say I don't know based on the provided data "
                    "Please don't provide any code just use the available data you have, in case the number doesn't match the data, don't try to fill in the gaps."
                )

                # Pass the base64 image directly to your pipeline
                self.vlm_cli.run(self.llm_model, user_question, b64_image)

            except CypherSyntaxError as e:
                self.get_logger().error(f"Cypher syntax error: {e}")
            except queue.Empty:
                # No image available; loop and check rclpy.ok()
                continue
            except Exception as e:
                self.get_logger().error(f"Error in VLM worker: {e}")


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
