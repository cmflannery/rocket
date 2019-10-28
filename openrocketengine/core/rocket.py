#!/usr/bin/env python
"""This module is part of Open Rocket Engine's engine development program. OpenRocketEngine
provides a command line interface for the design and development of rocket engine thrust
chambers."""
from __future__ import division, absolute_import, print_function
import sys
import os
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import xlsxwriter
from string import ascii_uppercase

__author__ = "Cameron Flannery"
__copyright__ = "Copyright 2018"
__license__ = "MIT"


np.warnings.filterwarnings('ignore')

SI_units = {
        'thrust': 'N',
        'mass': 'kg',
        'isp': 's',
        'mdot': 'kg/s',
        'unitless': '1',
        'length': 'm',
        'area': 'm^2',
        'volume': 'm^3',
        'angle': 'degrees'
        }

imperial_units = {
        'thrust': 'lbf',
        'mass': 'lbm',
        'isp': 's',
        'mdot': 'lbm/s',
        'unitless': '1',
        'length': 'ft',
        'area': 'ft^2',
        'volume': 'ft^3',
        'angle': 'degrees'
        }
Units = dict(SI=SI_units, Imperial=imperial_units)

# engine class retrieves and stores all outputs for each run
class Engine():
    """Designs a rocket engine based on isentropic flow equations.

    Initialization requirements are listed in the parameters below.

    Parameters:
        name (str): Common name for the engine
        thrust (float): sea-level thrust
        Tc (float): chamber temperature
        pc (float): chamber pressure
        pe (float): exit pressure
        pa (float): ambient pressure
        MR (float): mass ratio
        MW (float): molecular weight
        gamma (float): ratio of coefficients of heat
    """
    def __init__(self, thrust=None, Tc=None, pc=None, pe=None, pa=None, MR=None, MW=None,
                 gamma=None, lstar=None, area_ratio=None, units=None, contraction_angle=None,
                 bell_length=None, name=None):
        self.units = units
        self.name = name

        self.thrust = thrust
        self.Tc = Tc
        self.pc = pc
        self.pe = pe
        self.pa = pa
        self.MR = MR
        self.MW = MW
        self.gamma = gamma
        # geometric parameters
        self.lstar = lstar
        if area_ratio is not None:
            self.contraction_area_ratio = area_ratio  # chamber contraction area ratio
        else:
            self.contraction_area_ratio = 5  # nondimensional

        if contraction_angle is not None:
            self.contraction_angle = contraction_angle  # in degrees
        else:
            self.contraction_angle = 60 # degrees

        if bell_length is not None:
            self.bell_length = bell_length
        else:
            self.bell_length = 0.8

        if self.pa is None:
            self.pa = self.pe

        # set constants
        if self.units == "SI":
            self.__Rbar = 8314  # [kJ/Kmol-K]
            self.__g0 = 9.81  # m/s^2
        elif self.units == "Imperial":
            self.__Rbar = 1545  # [ft-lbf/lbmol-R]
            self.__g0 = 32.17  # ft/s^2


    @property
    def name(self):
        """ name, engine name: optional parameter """
        return self.__name

    @name.setter
    def name(self,value):
        """ set name """
        self.__name = value

################################################################################
#**************************** INDEPENDENT VARIABLES ****************************
################################################################################
    @property
    def Rbar(self):
        """ Rbar, universal gas constant """
        return self.__Rbar

    @property
    def g0(self):
        """ g0, gravitational parameter """
        return self.__g0

    @property
    def thrust_vac(self):
        """ Thrust in vacuum """
        self.__thrust_vac = self.thrust + self.pe*self.Ae
        return self.__thrust_vac

    @property
    def Tc(self):
        """ Tc, chamber temperature property """
        return self.__Tc

    @Tc.setter
    def Tc(self, value):
        self.__Tc = value

    @property
    def pc(self):
        """ pc, chamber pressure property """
        return self.__pc

    @pc.setter
    def pc(self, value):
        self.__pc = value

    @property
    def pa(self):
        """ pa, ambient pressure property """
        return self.__pa

    @pa.setter
    def pa(self, value):
        self.__pa = value

    @property
    def pe(self):
        """ pe, exit pressure proprety """
        return self.__pe

    @pe.setter
    def pe(self, value):
        self.__pe = value

    @property
    def MR(self):
        """" MR, mixture ratio property """
        return self.__MR

    @MR.setter
    def MR(self, value):
        self.__MR = value

    @property
    def MW(self):
        """ MW, gas molecular weight of propellants """
        return self.__MW

    @MW.setter
    def MW(self, value):
        self.__MW = value

    @property
    def gamma(self):
        """ gamma, ratio of coefficients of heats """
        return self.__gamma

    @gamma.setter
    def gamma(self, value):
        self.__gamma = value

################################################################################
#***************************** DEPENDENT VARIABLES *****************************
################################################################################
    @property
    def asound(self):
        """ asound is the speed of sound """
        Rspecific = self.Rspecific
        gamma = self.gamma
        Tc = self.Tc
        return np.sqrt(gamma*Rspecific*Tc)

    @property
    def Rspecific(self):
        """ Rspecific, specific gas constant """
        return self.Rbar/self.MW

    @Rspecific.setter
    def Rspecific(self, MW):
        self.__Rspecific = self.Rbar/MW

    @property
    def cstar(self):
        """ cstar, characteristic velocity
            get_cstar() calculates and returns the value of cstar """
        gamma = self.gamma
        Rspecific = self.Rspecific
        Tc = self.Tc
        return np.sqrt(gamma*Rspecific*Tc)/(gamma*np.sqrt((2/(gamma+1))**((gamma+1)/(gamma-1))))

    @cstar.setter
    def cstar(self, value):
        """ value, dictionary containing gamma, Rspecific, and Tc """
        g0 = self.g0
        required = ['gamma', 'Rspecific', 'Tc']

        try:
            gamma = value['gamma']
            Rspecific = value['Rspecific']
            Tc = value ['Tc']
        except KeyError:
            print('Include all required parameters:', required)
            raise

        self.__cstar = \
                np.sqrt(gamma*Rspecific*Tc)/(gamma*np.sqrt((2/(gamma+1))**((gamma+1)/(gamma-1))))

    @property
    def Cf(self):
        """ Cf, Thrust Coefficient """
        gamma = self.gamma
        pc = self.pc
        pe = self.pe
        return np.sqrt((2*gamma**2/(gamma-1))* \
                       (2/(gamma+1))**((gamma+1)/(gamma-1))* \
                       (1-(pe/pc)**((gamma-1)/gamma)))

    @Cf.setter
    def Cf(self, value):
        """ Cf, Thrust Coefficient """
        required = ['pe','pc','gamma']
        try:
            pe = value['pe']
            pc = value['pc']
            gamma = value['gamma']
        except KeyError:
            print('Include all required parameters:', required)
            raise

        self.__Cf = np.sqrt((2*gamma**2/(gamma-1))*(2/(gamma+1))**((gamma+1)/(gamma-1)) * \
                    (1-(pe/pc)**((gamma-1)/gamma)))

    @property
    def Isp(self):
        """ Isp, Specific Impulse [s]

        Specific Impulse is a commonly used performance metric for rocket
        engines
        """
        cstar = self.cstar
        Cf = self.Cf
        g0 = self.g0
        return cstar*Cf/g0

    @property
    def Isp_vac(self):
        """ Isp_vac, Specific Impulse in vacuum [s]
        """
        thrust = self.thrust + (self.pe-0)*self.Ae
        return thrust/(self.mdot*self.g0)

    @property
    def mdot(self):
        """ mdot, total mass flow rate

        typically,
            get_mdot(Isp=Isp, thrust=thrust, g0=g0)"""
        thrust = self.thrust
        Isp = self.Isp
        g0 = self.g0
        return thrust/(Isp*g0)

    @property
    def mdot_ox(self):
        """ mdot_ox, mass flow rate of oxidizer"""
        MR = self.MR
        mdot = self.mdot
        return MR/(MR+1)*mdot

    @property
    def mdot_f(self):
        """ mdot_f, mass flow rate of fuel"""
        MR = self.MR
        mdot = self.mdot
        return 1/(MR+1)*mdot

    @property
    def Tt(self):
        """ Calculate the throat temperature

        Derived from isentropic flow-critical temperature ratio """

        return self.Tc/(1 + (self.gamma-1)/2)

    @property
    def pt(self):
        """ Calculate the throat pressure

        Derived from the isentropic flow-critical pressure ratio """
        pc = self.pc
        gamma = self.gamma
        return pc * (2/(gamma+1))**(gamma/(gamma-1))

    @property
    def ue(self):
        """ calculate the exhaust velocity """

        gamma = self.gamma
        Rspecific = self.Rspecific
        Tc = self.Tc
        pc = self.pc
        pe = self.pe
        self.__ue = np.sqrt(2*gamma*Rspecific/(gamma-1)*Tc*(1-(pe/pc)**((gamma-1)/gamma)))
        return self.__ue


    @property
    def Ma_exit(self):
        """ Calculate and set the mach number at nozzle exit

        derived from pressure ratio equation"""
        # ue = self.ue
        # asound = self.asound
        # self.__Ma_exit  = ue/asound
        # return self.__Ma_exit
        gamma = self.gamma
        pc = self.pc
        pa = self.pa
        return np.sqrt(2/(gamma-1)*((pc/pa)**((gamma-1)/gamma)-1))

################################################################################
#**************************** Chamber Calculations *****************************
################################################################################

    # Notes:
    # Determine the area contraction ratio based on the injection velocities and
    # compressible flow equations in a converging section.
    # i.e. what convergence is necessary to achieve mach 1 in the throat?
    # This is codable.. Add some margin for major losses

    @property
    def Ac(self):
        """ Ac returns the chamber area """
        return self.At*self.contraction_area_ratio

    @property
    def Rc(self):
        """ radius of combustion chamber """
        return np.sqrt(self.Ac/np.pi)

    @property
    def Dc(self):
        """ diameter of combustion chamber """
        return 2*self.Rc

    @property
    def At(self):
        """ At returns the calculated area of the throat """
        mdot = self.mdot
        cstar = self.cstar
        pc = self.pc
        return cstar * mdot / pc

    @property
    def Rt(self):
        """ radius of the throat """
        return np.sqrt(self.At/np.pi)

    @property
    def Dt(self):
        """ diameter of the throat """
        return self.Rt*2

    @property
    def Ae(self):
        """ Ae returns the exit area of the rocket, assuming ideal expansion
        in a non-vacuum environment """
        return self.calc_A(self.Ma_exit)

    @property
    def Re(self):
        """ radius of exit """
        return np.sqrt(self.Ae/np.pi)

    @property
    def De(self):
        """ diameter of exit """
        return 2*self.Re

    @property
    def Rn(self):
        """ radius of circular entrance region for parabolic approximation (rao)
        i.e. region leaving the throat """
        return 0.382*self.Rt  # from Huzel and Huang, 76

    @property
    def R1(self):
        """ radius of circular entrance region to throat """
        return 1.5*self.Rt  # from rao and Braeuing

    @property
    def expansion_area_ratio(self):
        """ returns the expansion area ratio, Aexit/Athroat """
        return self.Ae/self.At

    @property
    def contraction_area_ratio(self):
        """ The contraction area ratio is the value of Ac/Ae. A minimum value of 3 is recommended
        to consistently achieve mach 1 in the throat """
        if self.__contraction_area_ratio:
            return self.__contraction_area_ratio
        else:
            self.__contraction_area_ratio = self.__Ac/self.__Ae
            if self.__contraction_area_ratio < 3:
                print("Warning: A minimum area contraction ratio of 3 is recommended. \
                      Use the Ae or contraction area ratio setter to change the value of Ac")
            return self.__contraction_area_ratio

    @contraction_area_ratio.setter
    def contraction_area_ratio(self,value):
        self.__contraction_area_ratio = value

    def calc_A(self, Ma):
        """ calc_A returns the area at an arbitrary station relative to the
        critical value, i.e. the throat in choked flow

        calc_A does not set any property of the class or class instance, only
        a value is returned """

        At = self.At
        gamma = self.gamma

        return At/Ma*((1 + ((gamma-1)/2)*Ma**2)/((1 + gamma)/2))**((gamma+1)/(2*(gamma-1)))

    @property
    def contraction_angle(self):
        return self.__contraction_angle

    @contraction_angle.setter
    def contraction_angle(self,value):
        """ set in degrees """
        self.__contraction_angle = value

    @property
    def lstar(self):
        return self.__lstar

    @lstar.setter
    def lstar(self, value):
        """ expects value in inches """
        self.__lstar = value

    @property
    def Vc(self):
        """ combustion chamber volume """
        return self.lstar*self.At

    @property
    def lcyl(self):
        """ length of cylindrical section of combustion chamber """
        Vc = self.Vc
        Ac = self.Ac
        Rc = self.Rc
        Rt = self.Rt
        beta = self.contraction_angle  # degrees

        return Vc/Ac - 1/2*(Rc-Rt)/np.tan(np.deg2rad(beta))

    @property
    def bell_length(self):
        """ the percentage length of a 15 degree conical nozzle

        default = 0.8 """

        return self.__bell_length

    @bell_length.setter
    def bell_length(self, value):
        self.__bell_length = value

    @property
    def ln(self):
        """ length of nozzle """
        Re = self.Re
        Rt = self.Rt
        bell_length = self.bell_length
        angle = 15.  # degrees
        return (Re-Rt)/np.tan(np.deg2rad(angle))*bell_length

    # Misc Tasks
    def generate_output(self):
        """Creates output excel file with engine performance and geometric parameters"""
        outputName = 'rocket_{name}_{now}.xlsx'.format(name=self.name, \
                                                       now=datetime.utcnow().strftime('%Y_%m_%d'))
        workbook = xlsxwriter.Workbook(outputName)
        geometryWorksheet = workbook.add_worksheet('geometry')
        performanceWorksheet = workbook.add_worksheet('performance')

        self._write_performance(performanceWorksheet)
        self._write_geometry(geometryWorksheet)

    def _write_performance(self, performanceWorksheet):
        """Write performance values to worksheet"""
        performanceWorksheet.write('A1', 'Engine Name:')
        if self.name != False:
            performanceWorksheet.write('B1', self.name)

        # generate header
        performanceWorksheet.write('A3', 'Thrust')
        performanceWorksheet.write('A4', 'Thrust Vac')
        performanceWorksheet.write('A5', 'Isp')
        performanceWorksheet.write('A6', 'Isp Vac')
        performanceWorksheet.write('A7', 'mass flow rate')
        performanceWorksheet.write('A8', 'Mixture Ratio')

        # add data
        performanceWorksheet.write('B3', self.thrust)
        performanceWorksheet.write('B4', self.thrust_vac)
        performanceWorksheet.write('B5', self.Isp)
        performanceWorksheet.write('B6', self.Isp_vac)
        performanceWorksheet.write('B7', self.mdot)
        performanceWorksheet.write('B8', self.MR)

        # units
        performanceWorksheet.write('C3', Units[self.units]['thrust'])
        performanceWorksheet.write('C4', Units[self.units]['thrust'])
        performanceWorksheet.write('C5', Units[self.units]['isp'])
        performanceWorksheet.write('C6', Units[self.units]['isp'])
        performanceWorksheet.write('C7', Units[self.units]['mdot'])
        performanceWorksheet.write('C8', Units[self.units]['unitless'])


    def _write_geometry(self, geometryWorksheet):
        """Write geometry values to worksheet"""
        geometryWorksheet.write('A1', 'Engine Name:')
        if self.name != False:
            geometryWorksheet.write('B1', self.name)

        # generate header
        geometryWorksheet.write('A3', 'Ac, Chamber Area')
        geometryWorksheet.write('A4', 'Rc, Chamber Radius')
        geometryWorksheet.write('A5', 'At, Throat Area')
        geometryWorksheet.write('A6', 'Rt, Throat Radius')
        geometryWorksheet.write('A7', 'Ae, Exit Area')
        geometryWorksheet.write('A8', 'Re, Exit Radius')
        geometryWorksheet.write('A9', 'Rn, radius leaving thoat')
        geometryWorksheet.write('A10', 'Ea, expansion area ratio (Ae/At)')
        geometryWorksheet.write('A11', 'Ec, contraction area ratio (Ac/Ae)')
        geometryWorksheet.write('A12', 'Thetac, contraction angle')
        geometryWorksheet.write('A13', 'lstar')
        geometryWorksheet.write('A14', 'Vc, chamber volume')
        geometryWorksheet.write('A15', 'lcyl, cylindrical section of combustion chamber')
        geometryWorksheet.write('A16', 'length of nozzle')

        # add data
        geometryWorksheet.write('B3', self.Ac)
        geometryWorksheet.write('B4', self.Rc)
        geometryWorksheet.write('B5', self.At)
        geometryWorksheet.write('B6', self.Rt)
        geometryWorksheet.write('B7', self.Ae)
        geometryWorksheet.write('B8', self.Re)
        geometryWorksheet.write('B9', self.Rn)
        geometryWorksheet.write('B10', self.expansion_area_ratio)
        geometryWorksheet.write('B11', self.contraction_area_ratio)
        geometryWorksheet.write('B12', self.contraction_angle)
        geometryWorksheet.write('B13', self.lstar)
        geometryWorksheet.write('B14', self.Vc)
        geometryWorksheet.write('B15', self.lcyl)
        geometryWorksheet.write('B16', self.ln)

        # units
        geometryWorksheet.write('C3', Units[self.units]['area'])
        geometryWorksheet.write('C4', Units[self.units]['length'])
        geometryWorksheet.write('C5', Units[self.units]['area'])
        geometryWorksheet.write('C6', Units[self.units]['length'])
        geometryWorksheet.write('C7', Units[self.units]['area'])
        geometryWorksheet.write('C8', Units[self.units]['length'])
        geometryWorksheet.write('C9', Units[self.units]['length'])
        geometryWorksheet.write('C10', Units[self.units]['unitless'])
        geometryWorksheet.write('C11', Units[self.units]['unitless'])
        geometryWorksheet.write('C12', Units[self.units]['angle'])
        geometryWorksheet.write('C13', Units[self.units]['length'])
        geometryWorksheet.write('C14', Units[self.units]['volume'])
        geometryWorksheet.write('C15', Units[self.units]['length'])
        geometryWorksheet.write('C16', Units[self.units]['length'])
