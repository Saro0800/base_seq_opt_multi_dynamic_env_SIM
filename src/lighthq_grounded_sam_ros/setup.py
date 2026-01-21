from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['prova_pkg_grounded_sam',
              'prova_pkg_grounded_sam.LightHQSAM',
              'prova_pkg_grounded_sam.segment_anything',
              'prova_pkg_grounded_sam.groundingdino'],
    package_dir={'': 'src'}
)

setup(**d)