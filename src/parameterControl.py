import numpy as np
from pathlib import Path
# ========== parameters
Ngrid = 40
force = 8.
timeLength = 20
dT = 0.05
intensedT = 0.01

dx = 1
isSave = True
noiseType = "Mixing"
noiseScale = 0.2
gaussianRatio = 0.5
observationOperatorType = "halfOBSOPT"
isCommonTruthInit = True
# ========== spin-up settings
initPerturb = 0.1
initSpingUpTime = 100.

# ========== don't touch these below (option controlling)
if noiseType != "Mixing":
    subFolderName = noiseType + "_" + str(noiseScale)
else:
    subFolderName = noiseType + str(gaussianRatio) + "_" +str(noiseScale)

# ========== time control
timeArray = np.arange(0, timeLength+dT, dT)
NtimeStep = len(timeArray)
intenseTimeArray = np.arange(0, timeLength+intensedT, intensedT)
NwindowSample = int(dT/intensedT) # used in 4DVar
intesneNtimeStep = len(intenseTimeArray)

parentDir = str(Path(__file__).parent.absolute())[:-3]