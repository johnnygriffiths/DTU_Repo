import numpy as np

import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# RegEx, creates markers/patterns to get only specific data frfom the files 
import re
                     
def CASCISimData(FilePath: str) -> pd.DataFrame:
    """
    Parse values for excitatoin energies and total oscillator strength from each excited state 
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
        r"\s+(?P<E>\d[.]\d{7,8})"
        )
    OscStrPattern = re.compile(
        r"\s+(?P<OscStr>\d+\.\d+[eE]?[-+]?\d+)"
    )
    # Store these in new list[dict]
    EnergData = []
    for Line in ExcELines:
        match = re.search(EnergPattern,Line)
        if match:
            EnergData.append(float(match.group('E'))
            )
    OscStrData = []
    for line in OscStrLines:
        match = re.search(OscStrPattern, line)
        if match:
            OscStrData.append(float(match.group('OscStr'))
            )
    
    #Store in dataframe
    SimDataDF = pd.DataFrame({"Excitation energy (a.u)": EnergData,"Total oscillator strength": OscStrData})

    return SimDataDF


FilePath = "./CAS46_lih.out"

CASCISimEnergiesDF = CASCISimData(FilePath)#, SkipRows)

# Number of roots set in Dalton
# Also corresponds to number of states reached
RootCount = 50
ExcitedStatesCount = np.arange(1,RootCount+1,1)

CASCISimEnergiesDF["Excited state no."] = ExcitedStatesCount
# Set the column to be the first one
CASCISimEnergiesDF = CASCISimEnergiesDF[ ["Excited state no."] + [col for col in CASCISimEnergiesDF.columns if col != "Excited state no."] ]
print(CASCISimEnergiesDF)