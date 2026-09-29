from glob import glob
from setuptools import find_packages, setup

setup(
    name='obstacle_detector_ros', version='5.1.0',
    packages=find_packages(include=['obstacle_detector*']),
    package_data={'obstacle_detector.route._core': ['rail_config.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/obstacle_detector_ros']),
        ('share/obstacle_detector_ros', ['package.xml', 'ROS2.md', 'VALIDATION_ROS2.md', 'requirements-humble.txt']),
        ('share/obstacle_detector_ros/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'], python_requires='>=3.10', zip_safe=False,
    maintainer='NIIstovye', maintainer_email='tactical-inventor@users.noreply.github.com',
    description='ROS 2 LiDAR loader and railway obstacle processor', license='Proprietary',
    entry_points={'console_scripts': [
        'lidar_loader = obstacle_detector_ros.loader_node:main',
        'obstacle_processor = obstacle_detector_ros.processor_node:main',
    ]},
)
