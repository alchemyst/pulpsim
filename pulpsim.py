#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy
import scipy.integrate
import matplotlib.pyplot as plt

# Simulate two comportments
# +----------+-----------+
# |          |           |
# |  liquor  |   wood    |
# |          |           |
# +----------+-----------+
#     x[0]       x[1]
#
# Initially, we have only diffiusion from the liquor to the wood
#

K = 0.1  # diffusion constant (mol/(m^2.s))
A = 1.1  # contact area (m^2)

liquor_volume = 1.0  # m^3
wood_volume = 1.0  # m^3

x0 = numpy.array([1, 0])

def dxdt(x, t):
    cl, cw = x

    transfer_rate = K*(cl - cw)

    # mass balance:
    return [-transfer_rate,
            +transfer_rate]

t = numpy.linspace(0, 100)

x, info = scipy.integrate.odeint(dxdt, x0, t, full_output=True)

plt.plot(t, x)
plt.show()
