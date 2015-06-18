#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy
import scipy.integrate
import matplotlib.pyplot as plt

# Simulate one liquor compartment and N wood compartments
# +----------+-----------+
# |          |   |   ... |
# |  liquor  |   wood    |
# |          | 0 | 1 ...N|
# +----------+-----------+
#            +-----------> z (spatial dimension)
#            0           1
# States are number of moles in each compartment
#     x[0]    x[1] | x[2] ...
#
# Initially, we have only diffusion from the liquor to the wood
#

t_end = 1000

K = 0.1  # diffusion constant (mol/(m^2.s))
A = 1.1  # contact area (m^2)
D = 0.001  # Fick's law constant
kr = 0.01  # reaction constant

liquor_volume = 1.0  # m^3
wood_volume = 1.0  # m^3
total_volume = liquor_volume + wood_volume

Ncompartments = 10
dz = 1./Ncompartments
wood_compartment_volume = wood_volume/Ncompartments

x0 = numpy.concatenate(([1], [0]*Ncompartments))

def dxdt(x, t):
    # unpack variables
    Nl = x[0]
    Nw = x[1:]

    # calculate concentrations
    cl = Nl/liquor_volume
    cw = Nw/wood_compartment_volume

    # All transfers are calculated in moles/second

    # Diffusion between liquor and first wood compartment
    transfer_rate = K*A*(cl - cw[0])

    # Flows for each block are due to diffusion
    #                                       v symmetry boundary
    #       +----+      +----+      +----+ ||
    # from  | 0  | d[0] | 1  | d[1] | 2  | ||
    # liq ->|    |----->|    |----->|    |-||
    #       +----+      +----+      +----+ ||


    # diffusion in wood (Fick's law)
    # The last compartment sees no outgoing diffusion due to symmetry
    gradcw = numpy.gradient(cw, dz)
    diffusion = -A*D*gradcw
    diffusion[-1] = 0

    # reaction rate in wood
    r = -kr*cw

    # mass balance:
    dNliquordt = -transfer_rate
    # in wood, we change due to diffusion (left and right) and reaction
    dNwooddt = r*wood_compartment_volume - diffusion + numpy.roll(diffusion, 1)
    # plus the extra flow from liquor
    dNwooddt[0] += transfer_rate

    return numpy.concatenate(([dNliquordt], dNwooddt))

def totalmass(x):
    return sum(x)

t = numpy.linspace(0, t_end)

xs, info = scipy.integrate.odeint(dxdt, x0, t, full_output=True)
C = xs/([liquor_volume] + [wood_compartment_volume]*Ncompartments)

# Concentrations
plt.subplot(2, 1, 1)
plt.plot(t, C)
plt.ylabel('Concentrations')
# Steady state should be
ss = sum(x0)/total_volume
plt.axhline(ss, color='black')
# Check that we aren't creating or destroying mass
plt.subplot(2, 1, 2)
plt.plot(t, [totalmass(x) for x in xs])
plt.ylabel('Total moles')
plt.ylim(ymin=0)
plt.show()
