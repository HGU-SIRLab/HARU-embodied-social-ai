from setuptools import find_packages, setup

package_name = 'haru_audio'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='herobot',
    maintainer_email='22100713@handong.ac.kr',
    description='HARU Audio Input Node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'audio_node = haru_audio.audio_node:main',
        ],
    },
)
