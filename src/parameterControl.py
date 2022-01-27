import numpy as np
from pathlib import Path

# >>>>>>>>>> parameters for lorenz96/lorenz96.py >>>>>>>>>>
Ngrid = 40       # number of spatial grids
force = 8.       # force in the equation
timeLength = 20  # entire time length
dT = 0.05        # analzing/forcasting time length
intensedT = 0.01 # designed for 4DVar needing intensive observations
# <<<<<<<<<< parameters for lorenz96/lorenz96.py <<<<<<<<<<

# >>>>>>>>>> parameters for dataGenerate/initValueGenerate.py >>>>>>>>>>
isSave = True
noiseType = "Mixing"
noiseScale = 0.2
gaussianRatio = 0.5
observationOperatorType = "halfOBSOPT"
isCommonTruthInit = True
# ========== spin-up settings
initPerturb = 0.1
initSpingUpTime = 100.
# <<<<<<<<<< parameters for dataGenerate/initValueGenerate.py <<<<<<<<<<

# >>>>>>>>>> global parameters >>>>>>>>>>
# ========== don't touch these below (option controlling)
parentDir = str(Path(__file__).parent.absolute())[:-3]
# ========== determine the subfolder name for data generated @ DAMEthod/data/
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
# <<<<<<<<<< global parameters <<<<<<<<<<

# >>>>>>>>>> I have no idea why they exist >>>>>>>>>>
dx = 1
# <<<<<<<<<< I have no idea why they exist <<<<<<<<<<



