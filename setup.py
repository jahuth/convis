from setuptools import setup
setup(
  name = 'convis',
  packages = ['convis','convis.filters','convis.patched_theano'],
  version = '0.3.0.4',
  install_requires = ["matplotlib", "Theano", "litus"],
  description = 'Convolutional Vision Model',
  author = 'Jacob Huth',
  author_email = 'jahuth@uos.de',
  url = 'https://github.com/jahuth/convis',
  download_url = 'https://github.com/jahuth/convis/tarball/0.3.0.3',
  keywords = ['vision model', 'retina model'],
  classifiers = [],
)
