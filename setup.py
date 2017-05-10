from setuptools import setup
setup(
  name = 'convis',
  packages = ['convis'],
  version = '0.2.1',
  install_requires = ["matplotlib", "Theano", "pygpu", "litus"],
  description = 'Convolutional Vision Model',
  author = 'Jacob Huth',
  author_email = 'jahuth@uos.de',
  url = 'https://github.com/jahuth/convis',
  download_url = 'https://github.com/jahuth/convis/tarball/0.2.1',
  keywords = ['vision model', 'retina model'],
  classifiers = [],
)
