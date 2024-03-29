from setuptools import setup, find_packages
import sandp

setup(
    name = "sandp",
    version = sandp.__version__,
    description = "processor for SanDix detector",
    author = "YWei, JYe",
    author_email = "ywei@physics.ucsd.edu, yejingqiang1992@gmail.com",
    url = "https://wiki.nigroup.ucsd.edu/doku.php?id=sandix",
    packages = ['sandp','sandp.config', 'sandp.data', 'sandp.plugin', 'sandp.smooth', 'sandp.findPoWa'],
    package_dir={'sandp': 'sandp'},
    package_data={'sandp': ['config/*.ini', 'smooth/*.so', 'findPoWa/*.so']},
    scripts=['bin/sandper']
)
