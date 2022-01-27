import numpy as np
from scipy.integrate import ode
import copy

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from parameterControl import *
from lorenz96.lorenz96 import Lorenz96

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
        Path(parentDir+"data/{}/initRecord/{}".format(observationOperatorType, subFolderName)).mkdir(parents=True, exist_ok=True)
        np.savetxt(parentDir+'data/{}/initRecord/{}/truthState.txt'.format(observationOperatorType, subFolderName), truthState)
        np.savetxt(parentDir+'data/{}/initRecord/{}/sparseTruthState.txt'.format(observationOperatorType, subFolderName), sparseTruthState)
        np.savetxt(parentDir+'data/{}/initRecord/{}/initAnalysisState.txt'.format(observationOperatorType, subFolderName), initAnalysisState)
        np.savetxt(parentDir+'data/{}/initRecord/{}/fullObservationState.txt'.format(observationOperatorType, subFolderName), fullObservationState)
        np.savetxt(parentDir+'data/{}/initRecord/{}/sparseObservationState.txt'.format(observationOperatorType, subFolderName), sparseObservationState)
        np.savetxt(parentDir+'data/{}/initRecord/{}/initEC.txt'.format(observationOperatorType, subFolderName), observationEC)
        np.savetxt(parentDir+'data/{}/initRecord/observationOperator.txt'.format(observationOperatorType), observationOperator)
        print("saved successfully in ./{}/initRecord/{}".format(observationOperatorType, subFolderName))

