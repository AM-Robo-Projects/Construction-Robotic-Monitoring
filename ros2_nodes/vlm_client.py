#!/usr/bin/env python3
"""
Simple client to test the VLM service.
Usage: 
  # Use ground truth position from odometry:
  ros2 run ros2_nodes vlm_client
  
  # Or specify custom position:
  ros2 run ros2_nodes vlm_client --ros-args -p x:=-4.231 -p y:=3.779 -p use_odom:=false
"""

import rclpy
from rclpy.node import Node
from ros2_nodes.srv import VLMQuery


class VLMClient(Node):
    def __init__(self):
        super().__init__('vlm_client')
        
        # Declare parameters
        self.declare_parameter('x', 0.0)  # 0.0 means use odometry
        self.declare_parameter('y', 0.0)  # 0.0 means use odometry
        self.declare_parameter('threshold', 3.0)
        self.declare_parameter('question', '')
        self.declare_parameter('use_odom', True)  # Use ground truth by default
        
        self.client = self.create_client(VLMQuery, 'vlm_query')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for VLM service...')
        
        self.get_logger().info('VLM service available!')
        
    def send_request(self):
        request = VLMQuery.Request()
        
        use_odom = self.get_parameter('use_odom').value
        
        if use_odom:
            # Send 0.0, 0.0 to signal using ground truth odometry
            request.robot_x = 0.0
            request.robot_y = 0.0
            self.get_logger().info('Requesting VLM query using ground truth position from /odom')
        else:
            request.robot_x = self.get_parameter('x').value
            request.robot_y = self.get_parameter('y').value
            self.get_logger().info(f'Requesting VLM query at custom position ({request.robot_x}, {request.robot_y})')
        
        request.threshold = self.get_parameter('threshold').value
        request.question = self.get_parameter('question').value
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info('='*60)
            self.get_logger().info(f'VLM Answer: {response.answer}')
            self.get_logger().info(f'Door Detected: {response.door_detected}')
            self.get_logger().info(f'Total Detections: {response.total_detections}')
            self.get_logger().info('='*60)
            return response
        else:
            self.get_logger().error('Service call failed')
            return None


def main(args=None):
    rclpy.init(args=args)
    client = VLMClient()
    client.send_request()
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
