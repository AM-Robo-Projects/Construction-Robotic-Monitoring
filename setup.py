from setuptools import setup
import os
from glob import glob

package_name = 'ros2_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files if you have any
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include config files if you have any
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maintainer',
    maintainer_email='you@example.com',
    description='Python utilities and scripts for BIM models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # VLM service node (main server)
            'ros_vlm_node = ros2_nodes.ros2_vlm:main',
            # VLM client (for testing)
            'vlm_client = ros2_nodes.vlm_client:main',
        ],
    },
)