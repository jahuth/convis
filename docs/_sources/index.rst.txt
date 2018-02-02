.. convis documentation master file, created by
   sphinx-quickstart on Wed Dec  7 18:52:26 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of Convis
***************************************

Convis lets you build vision models using PyTorch.


Installation
------------

Installing `convis` and `PyTorch` itself is not complicated.
Go to http://pytorch.org and follow the installation instructions, either for `pip` or `conda`.

Requirements for the base installation are: Python 2.7 or Python 3.6 and PyTorch. All other requirements are installed when running::

   pip install convis

or for the most recent version::

   pip install git+https://github.com/jahuth/convis.git


I recommend installing opencv, and jupyter notebook, if you do not already have it installed::

   pip install convis notebook
   # eg. for ubuntu:
   sudo apt-get install python-opencv


Contents:

.. toctree::
   :maxdepth: 3

   installation
   usage
   examples
   filters
   filters_Conv3d
   pytorch_basics
   model
   docs


Found a bug or want to contribute?
----------------------------------

Bug reports and feature requests are always welcome! 
The best place to put them is the `github issue tracker <https://github.com/jahuth/convis/issues>`_.
If you have questions about usage of functions and classes and can not find 
an answer in the documentation and docstrings, this is considered a bug and I appreciate
it if you open an issue for that!

If you want, you can flag your issue already with one of the labels:

 * `bug <https://github.com/jahuth/convis/labels/bug>`_
 * `enhancement <https://github.com/jahuth/convis/labels/enhancement>`_
 * `missing documentation <https://github.com/jahuth/convis/labels/missing%20documentation>`_
 * `question <https://github.com/jahuth/convis/labels/question>`_

If you have fixed a bug or added a feature and you would like to see the change
included in the main repository, the preferred method is for you to commit the 
change to a fork on your own github account and submit a pull request.


General discussion is encouraged on the two mailing lists:
 * `convis-users@googlegroups.com <https://groups.google.com/forum/#!forum/convis-users>`_ for announcements and user questions
 * `convis-dev@googlegroups.com <https://groups.google.com/forum/#!forum/convis-dev>`_ for discussions about the development





Indices
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

