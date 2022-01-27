import numpy as np
from scipy.integrate import ode
from matplotlib.pyplot import *
import copy
import pathlib
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

class dataGenerator:
    def __init__(self, NtimeStep):
        self.Ngrid = Ngrid
        self.initPerturb = initPerturb
        self.initSpingUpTime = initSpingUpTime
        self.NtimeStep = NtimeStep

    def setupLorenz96(self, Nstep):
        xInit = self.initPerturb * np.random.randn(self.Ngrid)
        self.lorenz96 = Lorenz96(xInit)
        self.lorenz96.solver = self.lorenz96.getODESolver(Nstep=Nstep)

    def getInitValue(self):
        self.setupLorenz96(Nstep=10000)
        xInit = self.lorenz96.solveODE(endTime=self.initSpingUpTime)
        return xInit

    def getSeriesTruth(self, initValue):
        xTruth = initValue # initial truth
        self.lorenz96 = Lorenz96(xTruth)
        solver = self.lorenz96.getODESolver()
        nowTimeStep = 1
        while solver.successful() and nowTimeStep <= self.NtimeStep-1: # exclude zero
            solver.integrate(solver.t + intensedT)
            xTruth = np.vstack([xTruth, [solver.y]])
            if nowTimeStep % 50 == 0:
                print("Current TimeStep: {NTS:03d} | Time: {ST}".format(NTS=nowTimeStep, ST=round(solver.t, 5)))
            nowTimeStep += 1
        return xTruth

    def getSeriesObs(self, xTruth, loc, scale, noiseType):
        XObservation = copy.deepcopy(xTruth)
        if noiseType == "Gaussian":
            for i in range(self.NtimeStep):
                noise = np.random.normal(loc, scale, size=(Ngrid, ))
                XObservation[i] += noise
        elif noiseType == "Laplace":
            for i in range(self.NtimeStep):
                noise = np.random.laplace(loc, scale, size=(Ngrid, ))
                XObservation[i] += noise
        elif noiseType == "Mixing":
            for i in range(self.NtimeStep):
                laplaceNoise = np.random.laplace(loc, scale, size=(int(Ngrid * (1 - gaussianRatio), )))
                gaussianNoise = np.random.normal(loc, scale, size=(int(Ngrid * (gaussianRatio), )))
                noise = np.hstack((laplaceNoise, gaussianNoise))
                XObservation[i] += noise
        elif noiseType == "None":
            pass
        return XObservation

    def sparseVar(self, var, skip=5):
        sparseVar = var[::skip]
        return sparseVar

if __name__ == "__main__":
    # ========== build truth
    stateGenerator = dataGenerator(NtimeStep=intesneNtimeStep)
    truthState = stateGenerator.getInitValue()
    truthState = stateGenerator.getSeriesTruth(initValue=truthState)
    # truthState shape: (1001, 40)
    sparseTruthState  = stateGenerator.sparseVar(truthState)
    # sparseTruthState shape: (201, 40)

    # ========= build analysis init state
    initAnalysisState = stateGenerator.getInitValue()
    # initAnalysisState shape: (40, )


    # ========= build observation
    fullObservationState = stateGenerator.getSeriesObs(xTruth = truthState, loc=0.0, scale=noiseScale, noiseType=noiseType)
    # fullObservationState shape: (1001, 40)
    observationEC = np.cov((fullObservationState - truthState).transpose())
    sparseObservationState = stateGenerator.sparseVar(fullObservationState)
    # sparseObservationState shape: (201, 40)

    # ========== Observation Operator for all DA method
    observationOperator = np.identity(Ngrid)
    if "half" in observationOperatorType:
        observationOperator = np.zeros((int(Ngrid/2), Ngrid))
        for i in range(int(Ngrid/2)):
            observationOperator[i, i*2] = 1
        fullObservationState = (observationOperator @ fullObservationState.transpose()).transpose()
        sparseObservationState = (observationOperator @ sparseObservationState.transpose()).transpose()


    if isSave:
        pathlib.Path("{}/initRecord/{}".format(observationOperatorType, subFolderName)).mkdir(parents=True, exist_ok=True)
        np.savetxt('{}/initRecord/{}/truthState.txt'.format(observationOperatorType, subFolderName), truthState)
        np.savetxt('{}/initRecord/{}/sparseTruthState.txt'.format(observationOperatorType, subFolderName), sparseTruthState)
        np.savetxt('{}/initRecord/{}/initAnalysisState.txt'.format(observationOperatorType, subFolderName), initAnalysisState)
        np.savetxt('{}/initRecord/{}/fullObservationState.txt'.format(observationOperatorType, subFolderName), fullObservationState)
        np.savetxt('{}/initRecord/{}/sparseObservationState.txt'.format(observationOperatorType, subFolderName), sparseObservationState)
        np.savetxt('{}/initRecord/{}/initEC.txt'.format(observationOperatorType, subFolderName), observationEC)
        np.savetxt('{}/initRecord/observationOperator.txt'.format(observationOperatorType), observationOperator)
        print("saved successfully in ./{}/initRecord/{}".format(observationOperatorType, subFolderName))

