# daMethod/variationalMethods.py
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
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

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










class increFourDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

    def getObservationFromWindow(self, observationState, tidx, NwindowSample=NwindowSample):
        """including head and tail"""
        sampleObservationState = observationState[tidx*NwindowSample:(tidx+1)*NwindowSample+1]
        return sampleObservationState

    #def costFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
    def costFunction(self, analysisIncrement, trajectoryM, innovation, backgroundEC, observationEC):
        H = self.obsOperator
        backgroundCost = (analysisIncrement).transpose() @ np.linalg.inv(backgroundEC) @ (analysisIncrement)

        totalCost = backgroundCost
        for i in range(NwindowSample+1): # including head and tail
            observationCost = (H @ trajectoryM[i] @ analysisIncrement - innovation[i]).transpose() @ \
                              np.linalg.inv(observationEC) @ (H @ trajectoryM[i] @ \
                              analysisIncrement - innovation[i])
            totalCost += observationCost
        return 0.5 * (totalCost)

    def gradientOfCostFunction(self, analysisIncrement, trajectoryM, innovation, backgroundEC, observationEC):
        H = self.obsOperator
        gradientBackgroundCost = np.linalg.inv(backgroundEC) @ (analysisIncrement)

        gradientTotalCost = gradientBackgroundCost
        for i in range(NwindowSample+1):
            gradientObservationCost = trajectoryM[i].transpose() @ H.transpose() @ np.linalg.inv(observationEC) @ (H @ trajectoryM[i] @ analysisIncrement - innovation[i])
            gradientTotalCost += gradientObservationCost
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
    def getAnalysisState(self, backgroundState, observationState, backgroundEC, observationEC, NouterLoop=1):
        guessState = backgroundState
        for outer in range(NouterLoop):
            guessState = self.outerLoop(guessState=guessState, \
                                        backgroundState = backgroundState, \
                                        observationState = observationState, \
                                        backgroundEC = backgroundEC, 
                                        observationEC = observationEC)
        return guessState

    def outerLoop(self, guessState, backgroundState, observationState, backgroundEC, observationEC, NinnerLoop=1):
        trajectoryM, trajectoryState = self.getTrajectoryState(initState=guessState)
        innovation = np.zeros((NwindowSample+1, int(Ngrid/2)))
        for i in range(NwindowSample+1):
            innovation[i] = observationState[i] - self.obsOperator @ trajectoryState[i]

        guessIncrement = guessState - backgroundState
        for inner in range(NinnerLoop):
            guessIncrement = self.innerLoop(guessIncrement = guessIncrement, \
                                               trajectoryM = trajectoryM, 
                                               innovation = innovation, \
                                               backgroundEC = backgroundEC, \
                                               observationEC = observationEC)
        guessState += guessIncrement
        return guessState

    def innerLoop(self, guessIncrement, trajectoryM, innovation, backgroundEC, observationEC):
        analysisIncrement = minimize(self.costFunction, x0=guessIncrement, \
                            args = (trajectoryM, innovation, backgroundEC, observationEC), \
                            method='CG', jac=self.gradientOfCostFunction).x
        return analysisIncrement