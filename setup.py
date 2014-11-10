from distutils.core import setup
import os
import numpy

setup(name="kmeans_radec", 
      url="https://github.com/esheldon/kmeans_radec",
      description="perform kmeans clustering on the unit sphere",
      packages=['kmeans_radec'],
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      install_requires=['numpy'],
      version="0.1")

