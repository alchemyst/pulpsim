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
#     x[N]  | x[N+1]  |
#       .    | ..
#     x[(N-1)*Nc]| x[NNc-1]
#
# We simulate a reaction
# 1 A -> 1 B
# r1 = kr1*Ca
# 1 B -> 1 C
# r2 = kr2*Cb
# dNdt = S*r*V

def reaction_rates(C, x, T):
    """ Calculate reaction rates for a column of component concentrations
    :param C:
    :return: reaction rates
    """

    CA, CB, CC = C
    Nl, Nw = unflatx(x)
    # Get total moles
    mass_frac = Nw.sum(axis=1)*componentsMM/wood_mass
    
    if mass_frac[2] >= phase_change_limit[0]:
        kr2 = 0.02
    elif mass_frac[2] >= phase_change_limit[1]:
        kr2 = 0.02
    else:
        kr2 = 0.02
    return numpy.array([kr1*CA,
                        kr2*CB])


def flatx(liquor, wood):
    """ Return a "flattened" version of the state variables """
    return numpy.concatenate((liquor[:, None], wood), axis=1).flatten()


def unflatx(x):
    """
    :param x: flattened state variables
    :return: liquor, wood state variables reshaped
    """
    rectangle = x.reshape((Ncomponents, Ncompartments+1))
    liquor = rectangle[:, 0]  # First column is liquor
    wood = rectangle[:, 1:]  # all the rest are wood
    return liquor, wood


def concentrations(x):
    """ Return concentrations given number of moles vector
    """
    Nl, Nw = unflatx(x)

    # calculate concentrations
    cl = Nl/liquor_volume
    cw = Nw/wood_compartment_volume

    return cl, cw


def temp(t):
    """ Temperature function
    """

    T = Ti + t * 0.1
    return T


def fick_constant(T):
    """ gives diffusion constant in (m^2/s)
    """
    D = kd*(T**0.5)*numpy.exp((-Ea)/(R*T))
    return D

components = ['A', 'B', 'C']
# Molar mass
componentsMM = [1., 1., 1.]
Ncomponents = len(components)
 # stoicheometric matrix, reagents negative, products positive
S = numpy.array([[-1, 1, 0],
                 [0, -1, 1]]).T
t_end = 100

Ti = 273.15  # (Kelvin)
R = 1
Ea = 1
phase_change_limit = numpy.array([0.5, 0.3])
K = numpy.array([0.1, 0.1, 0])  # diffusion constant (mol/(m^2.s))
A = 1.1  # contact area (m^2)
# FIXME: K and D should be specified in a similar way
kd = numpy.array([[0.0001], [0.0002], [0]])
kr1 = 0.01 # reaction constant (mol/(s.m^3))

wood_mass = 1.0  # kg
liquor_volume = 1.0  # m^3
wood_volume = 1.0  # m^3
total_volume = liquor_volume + wood_volume

Ncompartments = 30
dz = 1./Ncompartments
wood_compartment_volume = wood_volume/Ncompartments


# Initial conditions
Nliq0 = numpy.array([1., 0., 0.])
         
Nwood0 = numpy.zeros((Ncomponents, Ncompartments))

x0 = flatx(Nliq0, Nwood0)


def dxdt(x, t):
    # assert numpy.all(x>=0)
    T = temp(t)
    D = fick_constant(T)

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
    r = numpy.apply_along_axis(reaction_rates, 0, cw, x, T)
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
z = numpy.linspace(0, 1, Ncompartments)
zl = numpy.array([-dz*2, 0])  # z-coords of liquor
Nt = len(t)

xs, info = scipy.integrate.odeint(dxdt, x0, t, full_output=True)

# Work out concentrations
# TODO: This is probably inefficient
cl, cw = map(numpy.array, zip(*map(concentrations, xs)))

# Concentrations
ax = None
cm = plt.get_cmap('cubehelix')
for i, component in enumerate(components):
    ax = plt.subplot(Ncomponents + 1, 1, i+1, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.pcolormesh(t, zl, numpy.atleast_2d(cl[:, i]), cmap=cm)
    plt.pcolormesh(t, z, cw[:, i, :].T, cmap=cm)
    plt.ylabel('[{}]'.format(component))

# Check that we aren't creating or destroying mass
plt.subplot(Ncomponents+1, 1, Ncomponents+1, sharex=ax)
plt.plot(t, [totalmass(x) for x in xs])
plt.ylabel('Total moles')
plt.ylim(ymin=0)
plt.subplots_adjust(hspace=0)
plt.show()
