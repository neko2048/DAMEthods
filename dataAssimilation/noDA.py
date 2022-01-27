import numpy as np
from scipy.integrate import ode
import copy
from scipy.optimize import minimize
from matplotlib.pyplot import *
from initValueGenerate import Lorenz96
from parameterControl import *
from dataRecorder import RecordCollector

class NODA:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

    # forecast
    def getForecastState(self, analysisState, nowT):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=nowT)
        backgroundState = lorenz.solveODE(endTime=nowT+dT)
        return backgroundState

    # analyzing
    def getAnalysisState(self, backgroundState):
        analysisState = backgroundState
        return analysisState


if __name__ == "__main__":
    # states (intense version is loaded)
    xInitAnalysis = np.loadtxt("{}/initRecord/{}/initAnalysisState.txt".format(observationOperatorType, subFolderName))
    xFullObservation = np.loadtxt("{}/initRecord/{}/sparseObservationState.txt".format(observationOperatorType, subFolderName))
    xTruth = np.loadtxt("{}/initRecord/{}/sparseTruthState.txt".format(observationOperatorType, subFolderName))

    # covariance 
    analysisEC = np.loadtxt("{}/initRecord/{}/initEC.txt".format(observationOperatorType, subFolderName))
    if "full" in observationOperatorType:
        observationEC = np.identity(Ngrid) * (noiseScale ** 2)
    else:
        observationEC = np.identity(int(Ngrid/2)) * (noiseScale ** 2)

    # collector
    dataRecorder = RecordCollector(methodName="noDA", noiseType=noiseType)
    if not dataRecorder.checkDirExists():
        dataRecorder.makeDir()

    # initial setup
    noDA = NODA(xInitAnalysis)
    noDA.analysisState = xInitAnalysis
    noDA.analysisEC = analysisEC
    noDA.observationState = xFullObservation[0]
    noDA.observationEC = observationEC
    noDA.forecastState = np.zeros(Ngrid)
    noDA.forecastEC = np.zeros((Ngrid, Ngrid))
    noDA.RMSE = np.sqrt(np.mean((noDA.analysisState - xTruth[0])**2))
    noDA.MeanError = np.mean(noDA.analysisState - xTruth[0])
    print("{:02f}: {:05f}".format(0, noDA.RMSE))
    dataRecorder.record(noDA, tidx=0)
    
    for tidx, nowT in enumerate(timeArray[:-1]):
        noDA.forecastState = noDA.getForecastState(noDA.analysisState, nowT=nowT)
        noDA.analysisState = noDA.getAnalysisState(backgroundState=noDA.forecastState)

        noDA.RMSE = np.sqrt(np.mean((noDA.analysisState - xTruth[tidx+1])**2))
        noDA.MeanError = np.mean(noDA.analysisState - xTruth[tidx+1])
        print("{:02f}: {:05f}".format(nowT+dT, noDA.RMSE))
        dataRecorder.record(noDA, tidx=tidx+1)
    dataRecorder.saveToTxt()
