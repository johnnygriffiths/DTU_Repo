import numpy as np

import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# RegEx, creates markers/patterns to get only specific data frfom the files 
import re
            
def ReadSimExcEnergiesFunc(FilePath: str, RootCount: int) -> int:
    
    # The line starts with Excitation'
    pattern = r'@ Excitation energy :'
    # Open the filename within the path
    with open(FilePath) as f:
        # Enumerate gives the index of that line
        for linecounter,line in enumerate(f):
            # When the pattern is found within file
            if re.match(pattern, line):
                # Return index of that line
                return linecounter
            
    
def CASCISimData(FilePath: str,SkipRows: int) -> list[dict]:
    """
    Simulation data
    """
    #Stores each line in output file
    with open(FilePath,"r") as f:
        TableLines = f.readlines()
    #Stores the lines after the amount specified by skiprows
    TableLines = TableLines[SkipRows:]
    # Obtain from each line the state number, the energy (Eh) and the excitation energy (eV)
    ExcELines = [Line for Line in TableLines if "@ Excitation energy :" in Line]
    OscStrLines =  [Line for Line in TableLines if "@ Total oscillator strength " in Line ]
    ExcELines = [elem.replace("@", "") for elem in ExcELines]
    OscStrLines = [elem.replace("@", "") for elem in OscStrLines]
 
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
    SimData = pd.DataFrame({"Excitation energy (a.u)": EnergData,"Total oscillator strength": OscStrData})

    return SimData


FilePath = "./CAS46_lih.out"
# Number of roots set in Dalton
# Also corresponds to number of states reached
RootCount = 50

SkipRows = ReadSimExcEnergiesFunc(FilePath, RootCount)
#print(SkipRows)
CASCISimEnergies = CASCISimData(FilePath, SkipRows)
CASCISimEnergiesDF = pd.DataFrame(CASCISimEnergies)
CASCISimEnergiesDF.index +=1
print(CASCISimEnergiesDF)