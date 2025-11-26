Getting Started
===============

Designing a Rocket Engine
-------------------------

What does Rocket Do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rocket will perform some basic calculations for rocket thrust chamber designs. Additionally,
it will be able to perform comparisions of different configurations.

This code was originally created with the intention of automating vehicle sizing trade studies.

Design Inputs
~~~~~~~~~~~~~

.. note:: The physics behind the parameters mentioned here are not discussed. For a discussion on the
          physics behind the code refer to literature or gbok-propulsion_.

The inputs are classified as either required or optional.

* :code:`name`: [str] The name of the engine; used for naming output files.
* :code:`units`: [str] (SI or Imperial)
* :code:`thrust`: [float] Engine thrust (duh)
* :code:`Tc`: [float] Chamber Temperature
* :code:`pc`: [float] Chamber Pressure
* :code:`pe`: [float] Exit Pressure
* :code:`MR`: [float] Mixture Ratio :math:`\frac{\dot{m}_{oxidizer}}{\dot{m}_{fuel}}`

Configuration Files
~~~~~~~~~~~~~~~~~~~
rocket takes a configuration file as the only input, specifying the engine propellant properties, pressures desired,
and geometric design choices. Right now, there is only one possible combination of parameters that all have to be included in
the config file. In the future, there may be additional options to automatically retrieve propellant properties from CEA.

Config files are usually named with the engine name and the revision number with a '.cfg' suffix. I.e. RBF-rev01.cfg.

A typical configuration file looks like the following::

    # This is a test configuration file for rocket
    #
    # The parameters listed here are all the known parameters that rocket can take as inputs.
    # Refer to the official documentation for more implementation and usage details.
    name RBF1
    units SI
    thrust 5000
    Tc 3200
    pc 2068000
    pe 101325
    MR 2.1
    MW 18.9
    gamma 2.31
    # Geometric parameters
    lstar 1.016
    area_ratio 5.5


Running the program
~~~~~~~~~~~~~~~~~~~
rocket can be fun from the command line with the command `rocket`::

    $ rocket RBF-rev01.cfg

Outputs
~~~~~~~
rocket generates an output excel workbook with two sheets; one geometric parameters, and one for performance parameters.


Recommended Workflow
~~~~~~~~~~~~~~~~~~~~
Create a directory for your enigne design files and run rocket from there. I.e.::

    ~/Documents/marginal-stability/engine/design$ rocket tsu-01.cfg

.. _gbok-propulsion: https://gbok.readthedocs.io/en/latest/rockets.html
