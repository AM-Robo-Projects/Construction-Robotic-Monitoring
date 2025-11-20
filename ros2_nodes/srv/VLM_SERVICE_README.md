# VLM Service Node - Efficient On-Demand Vision-Language Model

## Overview
This package provides an efficient service-based architecture for querying a Vision-Language Model (VLM) with camera images and BIM knowledge graph data. The VLM is only initialized when a service request is made and shut down after processing to conserve resources.

## Features
- **Service-based triggering**: VLM only runs when explicitly requested
- **Resource efficient**: VLM pipeline shuts down after each query
- **Door detection counter**: Tracks positive door detections across queries
- **Dynamic robot position**: Query with current robot position and threshold
- **Latest frame caching**: Always uses most recent camera image

## Service Definition

### VLMQuery Service
**Request:**
- `string question` - Custom question (optional, uses default if empty)
- `float64 robot_x` - Robot X position in meters
- `float64 robot_y` - Robot Y position in meters  
- `float64 threshold` - Distance threshold in meters for "near" detection

**Response:**
- `string answer` - Full VLM response text
- `bool door_detected` - True if VLM detected a door (YES answer)
- `int32 total_detections` - Cumulative count of positive detections

## Build Instructions

```bash
# From workspace root
cd /home/grass/BIM_Navigation

# Source ROS2
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select ros2_nodes

# Source the workspace
source install/setup.bash
```

## Usage

### 1. Start the VLM Service Node
```bash
# Terminal 1: Start the service
ros2 run ros2_nodes ros_vlm_node
```

### 2. Call the Service

**Using the provided client:**
```bash
# Terminal 2: Call with default position
ros2 run ros2_nodes vlm_client

# Or with custom position
ros2 run ros2_nodes vlm_client --ros-args -p x:=-4.231 -p y:=3.779 -p threshold:=2.0
```

**Using ROS2 service CLI:**
```bash
# Call the service directly
ros2 service call /vlm_query ros2_nodes/srv/VLMQuery "{robot_x: -4.231, robot_y: 3.779, threshold: 2.0, question: ''}"
```

**Using Python:**
```python
import rclpy
from rclpy.node import Node
from ros2_nodes.srv import VLMQuery

# ... create node and client ...
request = VLMQuery.Request()
request.robot_x = -4.231
request.robot_y = 3.779
request.threshold = 2.0
request.question = ""  # Use default question

future = client.call_async(request)
# ... wait and process response ...
```

## Configuration

### Camera Topic
Edit `ros2_vlm.py` line 29 to change camera source:
```python
self.create_subscription(Image, "/your/camera/topic", self.image_callback, 10)
```

### Neo4j Connection
Edit `ros2_vlm.py` in `initialize_vlm_pipeline()` method:
```python
self.vlm_cli = VLMGraphRAGPipeline(
    neo4j_uri="neo4j://127.0.0.1:7687",
    neo4j_auth=("neo4j", "YOUR_PASSWORD"),
    llm_model=self.llm_model
)
```

### VLM Model
Edit `ros2_vlm.py` line 32:
```python
self.llm_model = "qwen3-vl:4b"  # Change to your preferred model
```

## Workflow

1. **Service starts**: Camera frames are cached but VLM is not loaded
2. **Service call received**: 
   - VLM pipeline initializes
   - Latest camera frame is retrieved
   - Query is processed with robot position and BIM data
   - VLM provides answer (YES/NO + explanation)
   - If "YES", detection counter increments
   - VLM pipeline shuts down to free memory
3. **Response returned**: Contains answer, detection status, and total count

## Benefits vs Continuous Mode

**Old continuous mode:**
- ❌ VLM runs constantly (high GPU/memory usage)
- ❌ Processes every camera frame unnecessarily
- ❌ No control over when queries run

**New service mode:**
- ✅ VLM only loads on demand
- ✅ Minimal resource usage when idle
- ✅ Precise control over query timing
- ✅ Counter tracks detections across navigation
- ✅ Easy integration with path planning

## Troubleshooting

**Service not available:**
```bash
ros2 service list | grep vlm_query
```

**Check if node is running:**
```bash
ros2 node list | grep vlm
```

**View logs:**
```bash
ros2 run ros2_nodes ros_vlm_node --ros-args --log-level debug
```

**No camera image:**
- Check camera topic is publishing: `ros2 topic hz /camera_face_camera/image_raw`
- Verify topic name matches in code

## Example Integration with Navigation

```python
# In your navigation node
from ros2_nodes.srv import VLMQuery

class NavigationNode(Node):
    def __init__(self):
        self.vlm_client = self.create_client(VLMQuery, 'vlm_query')
        
    def check_for_door_at_waypoint(self, x, y):
        request = VLMQuery.Request()
        request.robot_x = x
        request.robot_y = y
        request.threshold = 2.0
        
        future = self.vlm_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().door_detected:
            self.get_logger().info(f'Door found at waypoint ({x}, {y})!')
            return True
        return False
```

## Files
- `ros2_vlm.py` - Main service node
- `vlm_client.py` - Test client
- `srv/VLMQuery.srv` - Service definition
- `VLM_SERVICE_README.md` - This file

## License
MIT
