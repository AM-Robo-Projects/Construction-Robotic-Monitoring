cmake_minimum_required(VERSION 3.5)
project(ros2_nodes)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)

# Generate services
rosidl_generate_interfaces(${PROJECT_NAME}
  "ros2_nodes/srv/VLMQuery.srv"
  DEPENDENCIES std_msgs
)

# Install Python executables  
install(PROGRAMS
  ros2_nodes/ros2_vlm.py
  ros2_nodes/vlm_client.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)

ament_package()
