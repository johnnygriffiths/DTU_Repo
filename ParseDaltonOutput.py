import numpy as np

import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# RegEx, creates markers/patterns to get only specific data frfom the files 
import re
                     
def CASCISimData(FilePath: str) -> pd.DataFrame:
    """
    Parse values for excitation energies and total oscillator strength from each excited state 
    generated in output file by Dalton
    """
    #Stores each line in output file
    with open(FilePath,"r") as f:
        TableLines = f.readlines()

    # Obtain from each line the state number, the energy (Eh) and the excitation energy (eV)
    ExcELines = [Line for Line in TableLines if "@ Excitation energy :" in Line]
    OscStrLines =  [Line for Line in TableLines if "@ Total oscillator strength " in Line ]
    ExcELines = [elem.replace("@", "") for elem in ExcELines]
    OscStrLines = [elem.replace("@", "") for elem in OscStrLines]

    #Get only the actual numbers
    EnergPattern = re.compile(
        r"\s+(?P<E>\d+\.\d+)"
        )
    OscStrPattern = re.compile(
        r"\s+(?P<OscStr>\d+\.\d+[eE]?[-+]?\d+)"
    )
    # Store these in new list[dict]
    EnergData = []
    for EnergyLine in ExcELines:
        match = re.search(EnergPattern, EnergyLine)
        if match:
            EnergData.append(float(match.group('E'))*27.2114
            )
    OscStrData = []
    for OscStrLine in OscStrLines:
        match = re.search(OscStrPattern, OscStrLine)
        if match:
            OscStrData.append(float(match.group('OscStr')) #if want specific d.p then ('%f%' % float(..))
            )
    #Store in dataframe
    SimDataDF = pd.DataFrame({"Excitation energy (eV)": EnergData,"Total oscillator strength": OscStrData})

    return SimDataDF

ActiveSpace = "107"
System = "h2o"
BasisSet = "pvdz"
FileName = f"CAS{ActiveSpace}_{System}_{BasisSet}"
FilePath = f"./DaltonOutputs/{FileName}.out"

CASCISimEnergiesDF = CASCISimData(FilePath)
# The index count is the same as the root set in the Dalton file, i.e the number of excited states
CASCISimEnergiesDF.index +=1
CASCISimEnergiesDF.to_latex(f"./FilteredOutput/{System}/{FileName}.tex", index=False, longtable=True)