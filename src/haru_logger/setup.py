from setuptools import setup

package_name = 'haru_logger'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyungmin Cho',
    maintainer_email='22100713@handong.ac.kr',
    description='HARU HITL Episode Logger',
    license='MIT',
    entry_points={
        'console_scripts': [
            'hitl_node = haru_logger.hitl_node:main',
        ],
    },
)
