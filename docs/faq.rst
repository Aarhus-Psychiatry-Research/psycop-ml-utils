FAQ
-------


How do I test the code and run the test suite?    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Augmenty comes with an extensive test suite. To run the tests, you should clone the repository, then build augmenty from the source. 
This will also install the required development dependencies and test utilities defined in the requirements.txt.


.. code-block::
   
   # install test dependencies
   poetry install

   python -m pytest


which will run all the test in the :code:`tests` folder.

Specific tests can be run using:

.. code-block::

   python -m pytest tests/desired_test.py


If you want to check code coverage you can run the following:

.. code-block::

   python -m pytest --cov=.


How is the documentation generated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Augmenty uses `sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to generate documentation. It uses the `Furo <https://github.com/pradyunsg/furo>`__ theme with custom styling.

To make the documentation you can run:

.. code-block::

  # install sphinx, themes and extensions
  poetry install

  # generate html from documentations

  make -C docs html
  

How do I cite this work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use this library in your research, we please cite:

.. code-block::
   
   @software{Martin_PSYCOP_machine_learning,
      author = {Martin, Bernstorff and Lasse, Hansen and Enevoldsen, Kenneth},
      title = {{PSYCOP machine learning utilities}},
      url = {https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils},
      version = {0.1.1}
   }
