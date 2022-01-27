# daMethod/noDA.py
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
from parameterControl import *
from lorenz96.lorenz96 import Lorenz96
from dataRecord.dataRecorder import RecordCollector
# <<<<<<<<<< imported packages <<<<<<<<<<






class NODA:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt(parentDir+"data/{}/initRecord/observationOperator.txt".format(observationOperatorType))

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