from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['lighthq_grounded_sam_ros',
              'lighthq_grounded_sam_ros.LightHQSAM',
              'lighthq_grounded_sam_ros.segment_anything',
              'lighthq_grounded_sam_ros.groundingdino'],
    package_dir={'': 'src'}
)

setup(**d)