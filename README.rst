***********
``ARCCnet``
***********

Active Region Cutout Classification Network

Installation
============

The recommended way to install ``arccnet`` is with using `pip`.

.. code:: bash

    $ pip install git+https://github.com/ARCAFF/ARCCnet.git

To develop ``arccnet``, first fork the repository, clone the forked repository to your local machine, and install ``arccnet`` in an editable mode using the following commands:

   .. code-block:: bash

      $ git clone https://github.com/<username>/ARCCnet.git
      $ cd ARCCnet
      $ pip install -e .[dev]

If you are developing with `zsh` instead of `bash`, the last line should be:

    .. code-block:: zsh

     pip install -e '.[dev]'

and to test your changes

   .. code-block:: zsh

      coverage run -m pytest --cov=arccnet --cov-report=html

For more detailed instructions, see our `contributing guide <https://github.com/ARCAFF/ARCCnet/blob/main/CONTRIBUTING.rst>`__.

Usage
=====

To download and process the arccnet data:

.. code:: shell

   python arccnet/data_generation/data_generation.py
   python arccnet/data_generation/mag_processing.py

Here is a quick example of importing arccnet:

.. code:: python

   >>> import arccnet
   >>> arccnet.__version__

Getting Help
============

For more information or to ask questions about ``arccnet`` or any other SunPy library, check out our `issues <https://github.com/ARCAFF/ARCCnet/issues>`__.


Acknowledging or Citing ``ARCCnet``
=================================

If you use ``arccnet`` in we would appreciate your `citing it in your publications <https://github.com/ARCAFF/ARCCnet/blob/main/CITATION.rst>`__.

Contributing
============

If you would like to get involved, start by joining having a look at our  `contributing guide <https://github.com/ARCAFF/ARCCnet/blob/main/CONTRIBUTING.rst>`__.
This will walk you through getting set up for contributing.

Code of Conduct
===============

When you are interacting with the community you are asked to follow our `Code of Conduct <https://github.com/ARCAFF/ARCCnet/blob/main/CODE_OF_CONDUCT.rst>`__.
