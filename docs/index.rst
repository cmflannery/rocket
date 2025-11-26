.. rocket documentation master file, created by
   sphinx-quickstart on Sun Jul 29 01:55:39 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rocket's documentation!
============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

Overview
--------

Rocket performs the calculations for simple analysis and design of rocket engines. For a
general overview of the philosophy behind designing rocket engines, refer to the `rocket propulsion
section`_ of General Body of Knowledge (GBOK).

Installation
------------

Rocket only supports python 3.5 and above. Functionality with other python releases is
untested and not guaranteed. Basic usage::

    $ rocket config_file.cfg

Basic Usage
-----------

For basic usage, refer to `getting_started`_.

Note: checkout "Nomenclature" below if you are unsure about any of the variable used above

These input parameters are typical of what is used in industry design of rocket engines. If you are
unfamiliar with this process, I recommend reading "Rocket Propulsion Elements" for basics, and
"Mechanics and Thermodynamics of Propulsion" for a more rigorous mathematical overview.

To get these input parameters, you have essentially three options:

1. NASA CEA

 #. free, but difficult to use
 #. generally trust worthy data

2. Braeuing (http://www.braeunig.us/space/comb.htm)

 #. super simple and fast
 #. data is based on STANJAN (older version of CEA)
 #. no access to actual data files, so plot interpretation can result in signifant errors

3. Testing

 #. challenging, but fun
 #. expensive and not really necessary given the resources above

Nomenclature
------------
Tc : chamber temperature
thrust : Thrust
pc : chamber pressure
pa : ambient pressure
MR : propellant mixture ratio
MW : propellant gas molecular weight
gamma : ratio of coefficient of heats


.. _`rocket propulsion section`: https://gbok.readthedocs.io/en/latest/rockets.html
.. 