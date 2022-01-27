import numpy as np
from scipy.integrate import ode
import copy
from scipy.optimize import minimize
from matplotlib.pyplot import *
from initValueGenerate import Lorenz96
from parameterControl import *
from dataRecorder import RecordCollector

class fourDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

    def getObservationFromWindow(self, observationState, tidx, NwindowSample=NwindowSample):
        """including head and tail"""
        sampleObservationState = observationState[tidx*NwindowSample:(tidx+1)*NwindowSample+1]
        return sampleObservationState

    def costFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
        _, trajectoryState = self.getTrajectoryState(initState=analysisState)
        H = self.obsOperator
        backgroundCost = (backgroundState - trajectoryState[0]).transpose() @ np.linalg.inv(backgroundEC) @ (backgroundState - trajectoryState[0])

        totalCost = backgroundCost
        for i in range(NwindowSample+1): # including head and tail
            innovation = observationState[i] - H @ trajectoryState[i]
            observationCost = (innovation).transpose() @ np.linalg.inv(observationEC) @ (innovation)
            totalCost += observationCost
        return 0.5 * (totalCost)

    def gradientOfCostFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
        trajectoryM, trajectoryState = self.getTrajectoryState(initState=analysisState)
        H = self.obsOperator
        gradientBackgroundCost = np.linalg.inv(backgroundEC) @ (trajectoryState[0] - backgroundState)

        gradientTotalCost = gradientBackgroundCost
        for i in range(NwindowSample+1):
            innovation = observationState[i] - H @ trajectoryState[i]
            gradientObservationCost = trajectoryM[i].transpose() @ H.transpose() @ np.linalg.inv(observationEC) @ (innovation)
            gradientTotalCost -= gradientObservationCost
        return gradientTotalCost

    # >>>>>>>>> TLM >>>>>>>>>>
    def forceODE(self, x, force=force):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force

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

    def getTrajectoryState(self, initState, NwindowSample=NwindowSample):
        trajectoryState = np.zeros((NwindowSample+1, len(initState)))
        trajectoryState[0] = initState
        ddT = dT / NwindowSample
        trajectoryM = np.zeros((NwindowSample+1, Ngrid, Ngrid))
        trajectoryM[0] = np.identity(Ngrid)
        for i in range(NwindowSample):
            F = self.getJacobianOfForceODE(xState = trajectoryState[i])
            L_i = np.identity(Ngrid) + ddT * F
            trajectoryState[i+1] = trajectoryState[i] + ddT * self.forceODE(x=trajectoryState[i])
            trajectoryM[i+1] = L_i @ trajectoryM[i]
        return trajectoryM, trajectoryState
    # <<<<<<<<<< TLM <<<<<<<<<<

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


if __name__ == "__main__":
    # states (intense version is loaded)
    xInitAnalysis = np.loadtxt("{}/initRecord/{}/initAnalysisState.txt".format(observationOperatorType, subFolderName))
    xFullObservation = np.loadtxt("{}/initRecord/{}/fullObservationState.txt".format(observationOperatorType, subFolderName))
    xTruth = np.loadtxt("{}/initRecord/{}/TruthState.txt".format(observationOperatorType, subFolderName))

    # covariance 
    analysisEC = np.loadtxt("{}/initRecord/{}/initEC.txt".format(observationOperatorType, subFolderName))
    if "full" in observationOperatorType:
        observationEC = np.identity(Ngrid) * (noiseScale ** 2)
    else:
        observationEC = np.identity(int(Ngrid/2)) * (noiseScale ** 2)
 
    # collector
    dataRecorder = RecordCollector(methodName="fourDVar", noiseType=noiseType)
    if not dataRecorder.checkDirExists():
        dataRecorder.makeDir()

    # initial setup
    fourDvar = fourDVar(xInitAnalysis)
    fourDvar.analysisState = xInitAnalysis
    fourDvar.analysisEC = analysisEC
    fourDvar.forecastState = xInitAnalysis # presumed
    fourDvar.forecastEC = analysisEC # presumed
    fourDvar.observationWindowState = fourDvar.getObservationFromWindow(xFullObservation, tidx=0)
    fourDvar.observationState = fourDvar.observationWindowState[0]
    fourDvar.observationEC = observationEC
    fourDvar.RMSE = np.sqrt(np.mean((fourDvar.analysisState - xTruth[0])**2))
    fourDvar.MeanError = np.mean(fourDvar.analysisState - xTruth[0])
    print("{:02f}: {:05f}".format(0, fourDvar.RMSE))
    dataRecorder.record(fourDvar, tidx=0)

    for tidx, nowT in enumerate(timeArray[:-2]):
        fourDvar.observationWindowState = fourDvar.getObservationFromWindow(xFullObservation, tidx=tidx+1)
        fourDvar.observationState = fourDvar.observationWindowState[0]
        fourDvar.truthState = fourDvar.getObservationFromWindow(xTruth, tidx=tidx+1)
        fourDvar.forecastState = fourDvar.getForecastState(fourDvar.analysisState, nowT=nowT)
        fourDvar.analysisState = fourDvar.getAnalysisState(backgroundState=fourDvar.forecastState, \
                                                           observationState=fourDvar.observationWindowState, \
                                                           backgroundEC=fourDvar.forecastEC, \
                                                           observationEC=fourDvar.observationEC)

        fourDvar.RMSE = np.sqrt(np.mean((fourDvar.analysisState - fourDvar.truthState[0])**2))
        fourDvar.MeanError = np.mean(fourDvar.analysisState - fourDvar.truthState[0])
        print("{:02f}: {:05f}".format(nowT+dT, fourDvar.RMSE))
        dataRecorder.record(fourDvar, tidx=tidx+1)
    dataRecorder.saveToTxt()
