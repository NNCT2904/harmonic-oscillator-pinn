import numpy as np
import torch

def harmonic_oscillator_solution(d, w0, t):

    ''' Analytical solution to a under-damped harmonic oscillator, I am just going to use the solution given above

    d - delta, the damping ratio,

    w0 - undamped angular frequency,

    t - time,

    '''

    assert d < w0 # check for undamped case

    w = np.sqrt(w0**2-d**2)

    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    x = exp*2*A*cos
    
    return x