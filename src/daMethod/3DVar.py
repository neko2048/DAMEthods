# daMethod/3DVar.py
# Fu-Sheng Kao
# >>>>>>>>>> imported packages >>>>>>>>>>
import numpy as np
from scipy.integrate import ode
import copy
from scipy.optimize import minimize

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from dataGenerate import initValueGenerate
from parameterControl import *
from dataRecord.dataRecorder import RecordCollector
# <<<<<<<<<< imported packages <<<<<<<<<<






class threeDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

    def costFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
        H = self.obsOperator
        backgroundCost = (backgroundState - analysisState).transpose() @ np.linalg.inv(backgroundEC) @ (backgroundState - analysisState)

        innovation = observationState - H @ analysisState
        observationCost = (innovation).transpose() @ np.linalg.inv(observationEC) @ (innovation)
        return 0.5 * (backgroundCost + observationCost)

    def gradientOfCostFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
        H = self.obsOperator
        gradientBackgroundCost = np.linalg.inv(backgroundEC) @ (analysisState - backgroundState)

        innovation = observationState - H @ analysisState
        gradientObservationCost = H.transpose() @ np.linalg.inv(observationEC) @ (innovation)
        return gradientBackgroundCost - gradientObservationCost

    # forecast
    def getForecastState(self, analysisState, nowT):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=nowT)
        backgroundState = lorenz.solveODE(endTime=nowT+dT)
        return backgroundState

    # analyzing
    def getAnalysisState(self, backgroundState, observationState, backgroundEC, observationEC):
        analysisState = minimize(self.costFunction, x0=backgroundState, \
                        args = (backgroundState, observationState, backgroundEC, observationEC), \
                        method='CG', jac=self.gradientOfCostFunction).x
        return analysisState










class increThreeDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt(parentDir+"data/{}/initRecord/observationOperator.txt".format(observationOperatorType))

    def costFunction(self, analysisIncrement, innovation, backgroundEC, observationEC):
        H = self.obsOperator
        backgroundCost = (analysisIncrement).transpose() @ np.linalg.inv(backgroundEC) @ (analysisIncrement)

        observationCost = (H @ analysisIncrement - innovation).transpose() @ np.linalg.inv(observationEC) @ (H @ analysisIncrement - innovation)
        return 0.5 * (backgroundCost + observationCost)

    def gradientOfCostFunction(self, analysisIncrement, innovation, backgroundEC, observationEC):
        H = self.obsOperator
        gradientBackgroundCost = np.linalg.inv(backgroundEC) @ (analysisIncrement)

        gradientObservationCost = H.transpose() @ np.linalg.inv(observationEC) @ (H @ analysisIncrement - innovation)
        return gradientBackgroundCost + gradientObservationCost

    # forecast
    def getForecastState(self, analysisState, nowT):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=nowT)
        backgroundState = lorenz.solveODE(endTime=nowT+dT)
        return backgroundState