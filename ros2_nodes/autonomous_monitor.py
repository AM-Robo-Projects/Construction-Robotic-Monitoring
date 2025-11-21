#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from ros2_nodes.srv import VLMQuery
from neo4j import GraphDatabase
import math
import time

class AutonomousMonitor(Node):
    def __init__(self):
        super().__init__('autonomous_monitor')
        
        # Neo4j Connection
        self.declare_parameter('neo4j_uri', 'neo4j://127.0.0.1:7687')
        self.declare_parameter('neo4j_user', 'neo4j')
        self.declare_parameter('neo4j_password', 'CRANE_HALL')
        
        uri = self.get_parameter('neo4j_uri').value
        user = self.get_parameter('neo4j_user').value
        password = self.get_parameter('neo4j_password').value
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.get_logger().info('Connected to Neo4j')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Neo4j: {e}')
            return

        # VLM Service Client
        self.vlm_client = self.create_client(VLMQuery, 'vlm_query')
        while not self.vlm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for VLM service...')
        self.get_logger().info('VLM service available')

        # Nav2 Navigator
        self.navigator = BasicNavigator()

    def fetch_targets(self):
        """Fetch monitoring targets from Neo4j."""
        target_types = ['IfcDoor', 'IfcWindow'] # Add more as needed
        query = f"""
        MATCH (n)
        WHERE any(label in labels(n) WHERE label IN {str(target_types)})
          AND n.position_x IS NOT NULL AND n.position_y IS NOT NULL
        RETURN n.ifcId as id, labels(n) as labels, n.position_x as x, n.position_y as y
        """
        targets = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                lbls = record['labels']
                etype = next((l for l in lbls if l in target_types), 'Element')
                # Convert BIM coordinates to Map coordinates if necessary
                # Assuming BIM (x, y) matches Map (x, y) or applying simple transform
                # Note: In ros2_vlm.py, we saw a swap: robot_x = -msg.y, robot_y = msg.x
                # This implies BIM X = -Map Y, BIM Y = Map X
                # So Map X = BIM Y, Map Y = -BIM X
                
                bim_x = float(record['x'])
                bim_y = float(record['y'])
                
                # Transform to Map Frame for Navigation
                map_x = bim_y
                map_y = -bim_x
                
                targets.append({
                    'id': record['id'],
                    'type': etype,
                    'map_x': map_x,
                    'map_y': map_y,
                    'bim_x': bim_x,
                    'bim_y': bim_y
                })
        return targets

    def run_mission(self):
        # Wait for Nav2 to be active
        self.navigator.waitUntilNav2Active()
        
        targets = self.fetch_targets()
        self.get_logger().info(f'Starting mission with {len(targets)} targets...')
        
        # Get initial robot pose (approximate or wait for AMCL)
        # For sorting, we can assume we start at (0,0) or use the first target if we don't have a pose.
        # Better: Get the current robot pose from tf or odometry if possible, 
        # but BasicNavigator doesn't expose getRobotPose directly in all versions.
        # We'll assume start at (0,0) for the first sort, or just pick the closest to (0,0).
        current_x = 0.0
        current_y = 0.0

        # Optimization: Sort targets by nearest neighbor
        sorted_targets = []
        unvisited = targets[:]
        
        while unvisited:
            # Find nearest target to current position
            nearest_target = None
            min_dist = float('inf')
            nearest_idx = -1
            
            for i, target in enumerate(unvisited):
                dist = math.sqrt((target['map_x'] - current_x)**2 + (target['map_y'] - current_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_target = target
                    nearest_idx = i
            
            if nearest_target:
                sorted_targets.append(nearest_target)
                current_x = nearest_target['map_x']
                current_y = nearest_target['map_y']
                unvisited.pop(nearest_idx)
        
        targets = sorted_targets
        self.get_logger().info('Targets sorted by nearest neighbor.')

        for i, target in enumerate(targets):
            self.get_logger().info(f"Navigating to Target {i+1}/{len(targets)}: {target['type']} ({target['id']})")
            
            # Create Pose
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = target['map_x']
            goal_pose.pose.position.y = target['map_y']
            goal_pose.pose.orientation.w = 1.0 # Default orientation
            
            # Go to pose
            self.navigator.goToPose(goal_pose)
            
            while not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()
                # Optional: print feedback
                
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info('Goal reached! Backing up to improve view...')
                self.navigator.backup(backup_dist=0.5, backup_speed=0.1, time_allowance=10)
                while not self.navigator.isTaskComplete():
                    pass

                self.get_logger().info('Backing up complete. Spinning to capture surroundings...')
                self.navigator.spin(spin_dist=6.28, time_allowance=10)
                while not self.navigator.isTaskComplete():
                    pass
                
                self.get_logger().info('Spin complete. Triggering VLM...')
                self.trigger_vlm(target)
            else:
                self.get_logger().warn(f'Navigation failed for target {target["id"]}')
                
    def trigger_vlm(self, target):
        request = VLMQuery.Request()
        # Pass the BIM coordinates for the query context
        request.robot_x = target['bim_x']
        request.robot_y = target['bim_y']
        request.threshold = 3.0
        
        etype = target.get('type', 'building element')
        request.question = (
            f"Analyze the construction progress at position ({target['bim_x']:.2f}, {target['bim_y']:.2f}). "
            f"Query the database for any {etype} or other elements within 3m. "
            f"Compare the database expectations with the visual evidence. "
            f"Explicitly state if the expected {etype} is missing, correctly installed, or if there are unexpected objects."
        )
        
        future = self.vlm_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(f"VLM Analysis Complete: {response.door_detected}")

    def destroy_node(self):
        self.driver.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousMonitor()
    try:
        node.run_mission()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
