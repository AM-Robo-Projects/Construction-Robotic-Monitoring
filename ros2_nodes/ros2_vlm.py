#!/usr/bin/env python3
import os
import time
import threading
import queue
import datetime

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from neo4j.exceptions import CypherSyntaxError

from BIM_processing.KG_to_VLM import VLMGraphRAGPipeline
from BIM_processing.utils.prompt_manager import PromptManager
import base64

# Import the custom service (will be generated after build)
from ros2_nodes.srv import VLMQuery

class MyNode(Node):
    def __init__(self):
        super().__init__('vlm_service_node')
        self.get_logger().info('VLM Service Node has been started!')

        # Image subscription to keep latest frame
        self.create_subscription(Image, "/camera_face_camera/image_raw", self.image_callback, 10)
        
        # Odometry subscription for ground truth position (p3d plugin)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        # Create service
        self.srv = self.create_service(VLMQuery, 'vlm_query', self.handle_vlm_query)

        self.bridge = CvBridge()
        self.llm_model = "llama3.2-vision:11b"

        # Keep only the latest frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Keep latest robot position from odometry
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.odom_lock = threading.Lock()
        
        # Counter for positive door detections
        self.door_detection_count = 0
        
        # VLM pipeline will be initialized on-demand
        self.vlm_cli = None
        
        # Initialize prompt manager
        try:
            self.prompt_manager = PromptManager()
            self.get_logger().info(f'Prompt manager loaded. Available prompts: {self.prompt_manager.list_available_prompts()}')
        except Exception as e:
            self.get_logger().error(f'Failed to load prompt manager: {e}')
            self.prompt_manager = None
        
        self.get_logger().info('VLM service ready. Call /vlm_query service to trigger VLM processing.')
        self.get_logger().info('Listening to /odom for ground truth position')

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
        Fast callback: Store latest frame for service requests.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = self.bgr_to_rgb(cv_image)
            base64_str = self.cv_image_to_jpeg_bytes(cv_image)

            # Store latest frame
            with self.frame_lock:
                self.latest_frame = base64_str

        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")
            import traceback; traceback.print_exc()

    def odom_callback(self, msg: Odometry):
        """
        Odometry callback: Store latest ground truth robot position from p3d plugin.
        """
        try:
            with self.odom_lock:
                #switch to match the BIM model coordinate system
                self.robot_x = -msg.pose.pose.position.y 
                self.robot_y = msg.pose.pose.position.x
            
            # Log position occasionally (every 100 messages to avoid spam)
            if not hasattr(self, '_odom_count'):
                self._odom_count = 0
            self._odom_count += 1
            
            if self._odom_count % 100 == 0:
                self.get_logger().info(f'Robot position: ({self.robot_x:.3f}, {self.robot_y:.3f})')
                
        except Exception as e:
            self.get_logger().error(f"Odometry callback failed: {e}")
            import traceback; traceback.print_exc()

    def initialize_vlm_pipeline(self):
        """Initialize VLM pipeline on-demand"""
        if self.vlm_cli is None:
            self.get_logger().info('Initializing VLM pipeline...')
            self.vlm_cli = VLMGraphRAGPipeline(
                neo4j_uri="neo4j://127.0.0.1:7687",
                neo4j_auth=("neo4j", "CRANE_HALL"),
                llm_model=self.llm_model
            )
            self.get_logger().info('VLM pipeline initialized.')
    
    def shutdown_vlm_pipeline(self):
        """Shutdown VLM pipeline to free resources"""
        if self.vlm_cli is not None:
            self.get_logger().info('Shutting down VLM pipeline...')
            # Close Neo4j connection if available
            if hasattr(self.vlm_cli, 'close'):
                self.vlm_cli.close()
            self.vlm_cli = None
            self.get_logger().info('VLM pipeline shut down.')

    def save_vlm_response(self, x, y, answer):
        """Save VLM response to a text file for reporting."""
        try:
            # Determine path to reports directory relative to this script
            # Script is in src/RoboMonitoring/ros2_nodes/ros2_vlm.py
            # We want src/RoboMonitoring/reports/vlm_responses.txt
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            report_dir = os.path.join(base_dir, 'reports')
            
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            file_path = os.path.join(report_dir, 'vlm_responses.txt')
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(file_path, 'a') as f:
                f.write(f"--- VLM Report Entry ---\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Robot Position: ({x:.3f}, {y:.3f})\n")
                f.write(f"Response:\n{answer}\n")
                f.write(f"------------------------\n\n")
                
            self.get_logger().info(f"Saved VLM response to {file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save VLM response: {e}")

    def handle_vlm_query(self, request, response):
        """
        Service callback: Process VLM query with latest frame.
        If robot position is not provided in request (0.0, 0.0), use ground truth odometry.
        """
        # Use ground truth position if not specified in request
        if request.robot_x == 0.0 and request.robot_y == 0.0:
            with self.odom_lock:
                query_x = self.robot_x
                query_y = self.robot_y
            self.get_logger().info(f'Using ground truth position from odometry: ({query_x:.3f}, {query_y:.3f})')
        else:
            query_x = request.robot_x
            query_y = request.robot_y
            self.get_logger().info(f'Using requested position: ({query_x:.3f}, {query_y:.3f})')
        
        try:
            # Check if we have a frame
            with self.frame_lock:
                if self.latest_frame is None:
                    response.answer = "No image available yet. Please wait for camera feed."
                    response.door_detected = False
                    response.total_detections = self.door_detection_count
                    return response
                
                frame_to_process = self.latest_frame
            
            # Initialize VLM pipeline
            self.initialize_vlm_pipeline()
            
            # Build prompt from configuration or use custom question
            if request.question:
                user_question = request.question
                self.get_logger().info('Using custom question from request')
            elif self.prompt_manager:
                # Use prompt from configuration
                user_question = self.prompt_manager.get_prompt(
                    'door_detection_default',
                    robot_x=query_x,
                    robot_y=query_y,
                    threshold=request.threshold
                )
                self.get_logger().info('Using prompt from configuration: door_detection_default')
            else:
                # Fallback to simple prompt if prompt manager failed to load
                user_question = f"Is there a door visible or within {request.threshold}m of robot position ({query_x:.3f}, {query_y:.3f})? Answer YES or NO first, then explain."
                self.get_logger().warn('Using fallback prompt (prompt manager not available)')

            self.get_logger().info(f'Processing VLM query...')
            
            # Call VLM pipeline
            vlm_answer = self.vlm_cli.run(
                self.llm_model, 
                user_question, 
                frame_to_process,
                robot_x=query_x,
                robot_y=query_y,
                threshold=request.threshold
            )
            
            # Parse answer for door detection (look for YES/NO in first part of answer)
            answer_str = str(vlm_answer).strip()
            answer_upper = answer_str.upper()
            
            # Get parsing keywords from config
            positive_keywords = []
            negative_keywords = []
            if self.prompt_manager:
                positive_keywords = self.prompt_manager.get_parsing_keywords('positive_detection')
                negative_keywords = self.prompt_manager.get_parsing_keywords('negative_detection')
            
            # More flexible parsing - check for YES or NO in the beginning
            door_detected = False
            if answer_upper.startswith('YES') or 'YES' in answer_upper[:50]:
                door_detected = True
            elif answer_upper.startswith('NO') or 'NO' in answer_upper[:50]:
                door_detected = False
            else:
                # Fallback: look for keywords throughout answer
                if any(keyword in answer_upper for keyword in positive_keywords):
                    door_detected = True
                elif any(keyword in answer_upper for keyword in negative_keywords):
                    door_detected = False
            
            if door_detected:
                self.door_detection_count += 1
                self.get_logger().info(f'Door detected! Total detections: {self.door_detection_count}')
            else:
                self.get_logger().info(f'No door detected at position ({query_x:.3f}, {query_y:.3f})')
            
            # Set response
            response.answer = answer_str
            response.door_detected = door_detected
            response.total_detections = self.door_detection_count
            
            # Save response to file
            self.save_vlm_response(query_x, query_y, answer_str)

            self.get_logger().info(f'VLM processing completed. Door detected: {door_detected}')
            
            # Shutdown VLM to free resources
            self.shutdown_vlm_pipeline()
            
        except CypherSyntaxError as e:
            self.get_logger().error(f"Cypher syntax error: {e}")
            response.answer = f"Error: Cypher syntax error - {str(e)}"
            response.door_detected = False
            response.total_detections = self.door_detection_count
            self.shutdown_vlm_pipeline()
            
        except Exception as e:
            self.get_logger().error(f"Error in VLM processing: {e}")
            import traceback; traceback.print_exc()
            response.answer = f"Error: {str(e)}"
            response.door_detected = False
            response.total_detections = self.door_detection_count
            self.shutdown_vlm_pipeline()
        
        return response



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
