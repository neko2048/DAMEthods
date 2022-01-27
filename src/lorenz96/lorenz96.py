import numpy as np
from scipy.integrate import ode
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from parameterControl import *

class Lorenz96:
    def __init__(self, initValue):
        self.initValue = initValue
        self.initSpingUpTime = initSpingUpTime
        self.force = force

    def forceODE(self, time, x, force):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force

    def getODESolver(self, initTime=0., solve_method='dopri5', Nstep=10000):
        if Nstep:
            solver = ode(self.forceODE).set_integrator(name=solve_method, nsteps=Nstep)
        else:
            solver = ode(self.forceODE).set_integrator(name=solve_method)
        solver.set_initial_value(self.initValue, t=initTime).set_f_params(self.force)
        return solver

    def solveODE(self, endTime):
        self.solver.integrate(endTime)
        xSolution = np.array(self.solver.y, dtype="f8")
        return xSolution