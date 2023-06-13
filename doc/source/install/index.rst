.. _install:

Install
=======

cvxreg supports Python 3 on Linux, macOS, and Windows. You can install the latest version of the package using pip:

pip
---

.. code-block:: bash

    pip install cvxreg

Solver options
--------------
cvxreg is built on top of `CVXPY <https://www.cvxpy.org/>`_, which supports a variety of open-source solvers `ECOS <https://www.embotech.com/ECOS>`_, `OSQP <https://osqp.org/>`_, and `SCS <http://github.com/cvxgrp/scs>`_. 
We use the default solver `ECOS <https://www.embotech.com/ECOS>`_ in cvxreg. To use a different solver, simply install the solver and specify the solver name in the ``solver`` argument of the model.

::

    model = models.CR(solver='OSQP')

Install with commercial solvers
-------------------------------
Many other solvers can be called by cvxreg if installed separately. See the table in `CVXPY <https://www.cvxpy.org/tutorial/advanced>`_ for a list of supported solvers.

To see which solvers are available in your machine, run:

.. code-block:: python
    from cvxreg import installed_solvers
    print(installed_solvers())

To use a commercial solver, you need to install the solver and specify the solver name in the ``solver`` argument of the model. For example, to use `MOSEK <https://www.mosek.com/>`_:

::
    model = models.CR(solver='mosek')