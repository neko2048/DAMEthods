import numpy as np
from scipy.integrate import ode

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from parameterControl import *
from lorenz96.lorenz96 import Lorenz96
from dataRecord.dataRecorder import RecordCollector











class ExtKalFilter:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

    def forceODE(self, x, force=force):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force

    # forecasting
    def getForecastState(self, analysisState, startTime):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=startTime)
        forecastState = lorenz.solveODE(endTime=startTime+dT)
        return forecastState

    def getForcastEC(self, analysisState, analysisEC):
        jacobianM = self.getJacobianOfMfromTransition(analysisState)
        forecastEC = jacobianM @ analysisEC @ jacobianM.transpose()
        return forecastEC

    def getJacobianOfForceODE(self, xState):
        """F(t_i)"""
        jacobianOfForceMaxtrix = np.zeros((xState.shape[0], xState.shape[0]), dtype=float)
        x_n = xState
        x_np1 = np.roll(xState, -1) # x_n+1
        x_nm1 = np.roll(xState, +1) # x_n-1
        x_nm2 = np.roll(xState, +2) # x_n-2
        for i in range(len(xState)):
            tempState = np.array([-x_nm1[i], x_np1[i] - x_nm2[i], -1, x_nm1[i]])
            jacobianOfForceMaxtrix[i, :4] = tempState
            jacobianOfForceMaxtrix[i] = np.roll(jacobianOfForceMaxtrix[i], -2+i)
        return jacobianOfForceMaxtrix

    def getJacobianOfMfromTransition(self, analysisState, Nsplit=10):
        tempState = analysisState
        ddT = dT / Nsplit
        M = np.identity(len(tempState))
        for i in range(Nsplit):
            F = self.getJacobianOfForceODE(xState = tempState)
            L_i = np.identity(Ngrid) + ddT * F
            tempState = tempState + ddT * self.forceODE(x=tempState)
            M = L_i @ M
        return M

    # analyzing
    def getAnalysisWeight(self, forecastState, forecastEC, observationEC):
        H = self.obsOperator
        inverseMatrix = np.linalg.inv(H @ forecastEC @ (H.transpose()) + observationEC)
        K = forecastEC @ (H.transpose()) @ (inverseMatrix)
        return K

    def getAnalysisState(self, forecastState, observationState, KalmanGain):
        H = self.obsOperator
        innovation = (observationState - H @ forecastState)
        analysisState = forecastState + (KalmanGain @ innovation)
        return analysisState

    def getAnalysisEC(self, forecastEC, KalmanGain, inflation=1):
        H = self.obsOperator
        analysisEC = (np.identity(Ngrid) - (KalmanGain @ H)) @ forecastEC
        analysisEC = inflation * analysisEC
        return analysisEC