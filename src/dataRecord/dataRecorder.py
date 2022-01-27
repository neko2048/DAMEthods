import pathlib
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from parameterControl import *











class RecordCollector:
    def __init__(self, methodName, noiseType):
        self.methodName = methodName
        self.noiseType = noiseType
        self.analysisState = np.zeros((NtimeStep, Ngrid))
        #self.analysisEC    = np.zeros((NtimeStep, Ngrid, Ngrid))
        self.forecastState = np.zeros((NtimeStep, Ngrid))
        #self.forecastEC    = np.zeros((NtimeStep, Ngrid, Ngrid))
        if "half" in observationOperatorType:
            self.observationState = np.zeros((NtimeStep, int(Ngrid/2)))
        elif "full" in observationOperatorType:
            self.observationState = np.zeros((NtimeStep, Ngrid))
        #self.observationEC = np.zeros((NtimeStep, Ngrid, Ngrid))
        self.RMSE          = np.zeros(NtimeStep)
        self.MeanError     = np.zeros(NtimeStep)

    def record(self, DAClass, tidx):
        self.analysisState[tidx] = DAClass.analysisState
        #self.analysisEC[tidx] = DAClass.analysisEC
        self.forecastState[tidx] = DAClass.forecastState
        #self.forecastEC[tidx] = DAClass.forecastEC
        self.observationState[tidx] = DAClass.observationState
        #self.observationEC[tidx] = DAClass.observationEC
        self.RMSE[tidx] = DAClass.RMSE
        self.MeanError[tidx] = DAClass.MeanError

    def checkDirExists(self):
        goalDir = parentDir+"data/{}/{}_record".format(observationOperatorType, self.methodName)
        return pathlib.Path(goalDir).is_dir()

    def makeDir(self):
        goalDir = parentDir+"data/{}/{}_record".format(observationOperatorType, self.methodName)
        pathlib.Path(goalDir).mkdir(parents=True, exist_ok=True)

    def saveToTxt(self):
        np.savetxt(parentDir+"data/{}/{}_record/{}_analysisState.txt".format(observationOperatorType, self.methodName, subFolderName), self.analysisState)
        np.savetxt(parentDir+"data/{}/{}_record/{}_forecastState.txt".format(observationOperatorType, self.methodName, subFolderName), self.forecastState)
        np.savetxt(parentDir+"data/{}/{}_record/{}_observationState.txt".format(observationOperatorType, self.methodName, subFolderName), self.observationState)
        np.savetxt(parentDir+"data/{}/{}_record/{}_RMSE.txt".format(observationOperatorType, self.methodName, subFolderName), self.RMSE)
        np.savetxt(parentDir+"data/{}/{}_record/{}_MeanError.txt".format(observationOperatorType, self.methodName, subFolderName), self.MeanError)
        print("Save successfully")