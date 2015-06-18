#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy
import scipy.integrate
import matplotlib.pyplot as plt

# Simulate one liquor compartment and N wood compartments
# there are Nc components
# +----------+----------------------------+
# | liquor   | wood                       |
# |          |                            |
# |          | 0        | 1      ..      N|
# +----------+----------------------------+
#            +--------------------------> z (spatial dimension)
#            0         dz                 1
# States are number of moles of each reagent in each compartment
#     x[0]   | x[1]    | x[2] ... x[N-1]
#     x[Nc]  | x[Nc+1]  |
#       .    | ..
#     x[(N-1)*Nc]| x[NNc-1]
#
# We simulate a reaction
# 1 A -> 1 B
# r = kr*Ca
# dNdt = S*r*V

def reaction_rates(C):
    """ Calculate reaction rates for a column of component concentrations
    :param C:
    :return: reaction rates
    """

    CA, CB = C
    return numpy.array([kr*CA])


def flatx(dNliquordt, dNwooddt):
    """ Return a "flattened" version of the state variables """
    return numpy.concatenate((dNliquordt[:, None], dNwooddt), axis=1).flatten()


def concentrations(x):
    """ Return concentrations given number of moles vector
    """
    
    N = x.reshape((Ncomponents, Ncompartments+1))
    Nl = N[:, 0]  # First column is liquor
    Nw = N[:, 1:]  # all the rest are wood

    # calculate concentrations
    cl = Nl/liquor_volume
    cw = Nw/wood_compartment_volume

    return cl, cw

components = ['A', 'B']
Ncomponents = len(components)
S = numpy.array([[-1, 1]]).T  # stoicheometric matrix

t_end = 100

K = 0.1  # diffusion constant (mol/(m^2.s))
A = 1.1  # contact area (m^2)
D = 0.0  # Fick's law constant
kr = 0.0  # reaction constant (mol/(s.m^3))

liquor_volume = 1.0  # m^3
wood_volume = 1.0  # m^3
total_volume = liquor_volume + wood_volume

Ncompartments = 3
dz = 1./Ncompartments
wood_compartment_volume = wood_volume/Ncompartments


# Initial conditions
Nliq0 = numpy.array([1.,   # A
                     0.])   # B
         
Nwood0 = numpy.array([[1/3, 0, 0],
                      [0, 0, 0]])
          
x0 = flatx(Nliq0, Nwood0)


def dxdt(x, t):
    # unpack variables
    cl, cw = concentrations(x)

    # All transfers are calculated in moles/second

    # Diffusion between liquor and first wood compartment
    transfer_rate = K*A*(cl - cw[:, 0])

    # Flows for each block are due to diffusion
    #                                       v symmetry boundary
    #       +----+      +----+      +----+ ||
    # from  | 0  | d[0] | 1  | d[1] | 2  | ||
    # liq ->|    |----->|    |----->|    |-||
    #       +----+      +----+      +----+ ||

    # diffusion in wood (Fick's law)
    # The last compartment sees no outgoing diffusion due to symmetry
    # FIXME: This calculates gradients for both dimensions
    _, gradcwz = numpy.gradient(cw, dz)
    diffusion = -A*D*gradcwz
    diffusion[:, -1] = 0

    # reaction rates in wood
    r = numpy.apply_along_axis(reaction_rates, 0, cw)
    # change in moles due to reaction
    reaction = S.dot(r)*wood_compartment_volume

    # mass balance for liquor:
    dNliquordt = -transfer_rate
    # in wood, we change due to diffusion (left and right) and reaction
    dNwooddt = reaction - diffusion + numpy.roll(diffusion, 1)
    # plus the extra flow from liquor
    dNwooddt[:, 0] += transfer_rate

    return flatx(dNliquordt, dNwooddt)

def totalmass(x):
    return sum(x)

t = numpy.linspace(0, t_end)
Nt = len(t)

xs, info = scipy.integrate.odeint(dxdt, x0, t, full_output=True)

# Work out concentrations
# TODO: This is probably inefficient
cl, cw = map(numpy.array, zip(*map(concentrations, xs)))

# Concentrations
for i, component in enumerate(components):
    plt.subplot(Ncomponents + 1, 1, i+1)
    plt.plot(t, cl[:, i])
    plt.plot(t, cw[:, i, :])
    plt.ylabel('[{}]'.format(component))
# Steady state should be
ss = sum(x0)/total_volume
plt.axhline(ss, color='black')
# Check that we aren't creating or destroying mass
plt.subplot(Ncomponents+1, 1, Ncomponents+1)
plt.plot(t, [totalmass(x) for x in xs])
plt.ylabel('Total moles')
plt.ylim(ymin=0)
plt.show()
