import numpy as np

import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# RegEx, creates markers/patterns to get only specific data frfom the files 
import re

"""
requires verbose=4 pySCF arg to show S**2 values
"""
            
def CASCISimData(FilePath: str) -> pd.DataFrame:
    """
    Full CI data from output file
    """
    #Stores each line in output file
    with open(FilePath,"r") as f:
        TableLines = f.readlines()

    for linecounter, line in enumerate(TableLines):
        if "CASCI converged" in line:
            SkipRows = linecounter
    # Filter only desired spin state
    CorrectSpinLines = [Line for Line in TableLines if "S^2 = 0.0000000" in Line]
    EnergLines = [Line for Line in TableLines[SkipRows:] if "State " in Line]
    OscStrLines = [Line for Line in TableLines[SkipRows:] if " dE " in Line]
    
    # Obtain from each line only the state number and energy (Eh)
    EnergPattern = re.compile(
        # energy
        r"State\s+(?P<state>\d+)\:\sdE\s\=\s(?P<E>\d+\.\d+)\seV\,")
    OscPattern = re.compile(
        # oscillator strength
        r"State\s+(?P<state>\d+)\:\sdE\s\=\s\d+\.\d+\seV\,\s+f\s*=\s+(?P<osc>\d+[.]\d+)"
    )
    CorrectSpinStates = []
    StatePattern = re.compile(r"CASCI state\s+(?P<state>\d+)")
    for Line in CorrectSpinLines:
        match = re.search(StatePattern, Line)
        if match:
            CorrectSpinStates.append(int(match.group('state')))
    CorrectSpinStates = set(CorrectSpinStates)
    # Store these in new list[dict]
    EnergData = []
    for EnergyLine in EnergLines:
        match = EnergPattern.search(EnergyLine)
        if match:
            state = int(match.group('state'))
            energy = float(match.group('E'))
            if state in CorrectSpinStates:
                EnergData.append(energy)

    OscStrData = []
    for Line in OscStrLines:
        match = OscPattern.search(Line)
        if match:
            state = int(match.group('state'))
            OscStr = float(match.group('osc'))
            if state in CorrectSpinStates:
                OscStrData.append(OscStr)
    print(len(OscStrData))
    print(len(EnergData))
    c = [x for x in EnergData if x not in OscStrData]
    print(c)
    #Store in dataframe
    SimDataDF = pd.DataFrame({"Excitation energy (eV)": EnergData,"Total oscillator strength": OscStrData})
    SimDataDF.index +=1
    return SimDataDF
    

"""
Get the first output from  pySCF (with S**2)
remove all the lines NOT containing S**2 = 0.000000

then ensure collect only these lines from the second output

then later get the same lines with the oscillator strength

"""

FilePath = "./pySCFsims/beh2_67.out"
CASCISimEnergiesDF = CASCISimData(FilePath)
print(CASCISimEnergiesDF)