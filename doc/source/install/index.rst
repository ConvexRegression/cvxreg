.. _install:

Install
=======

cvxreg supports Python 3 on Linux, macOS, and Windows. You can install the latest version of the package using pip:

pip
---

.. code-block:: bash

    pip install cvxreg

Install with MOSEK support
--------------------------
cvxreg supports the MOSEK solver.
Simply install MOSEK such that you can ``import mosek`` in Python.
See the `MOSEK <https://www.mosek.com/resources/getting-started/>`_ website for installation instructions.

Install with Gurobi support
---------------------------
cvxreg supports the Gurobi solver.
Simply install Gurobi such that you can ``import gurobipy`` in Python.
See the `Gurobi <https://www.gurobi.com/>`_ website for installation instructions.

Install with CPLEX support
--------------------------
cvxreg supports the CPLEX solver.
Simply install CPLEX such that you can ``import cplex`` in Python.
See the `CPLEX <https://www.ibm.com/analytics/cplex-optimizer>`_ website for installation instructions.

Install with COPT support
-------------------------
cvxreg supports the COPT solver.
Simply install COPT such that you can ``import copt`` in Python.
See the `COPT <https://github.com/COPT-Public/COPT-Release>`_ release page for installation instructions.

Install without local solvers
-----------------------------
cvxreg can also be installed without any local solvers.
cvxreg can interface with Pyomo to access the network-enabled optimization system `NEOS <https://neos-server.org/neos/>`_ solver. 
Please note that this free solver is not always available and may have a long queue time.
To use remote solvers, specify ``email`` and ``solver`` in the models. For example:

::

    model = models.CR(email='email@address', solver='knitro')