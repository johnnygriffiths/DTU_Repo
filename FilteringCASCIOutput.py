import numpy as np

import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# RegEx, creates markers/patterns to get only specific data frfom the files 
import re

def ReadDesiredEnergiesFunc(FilePath):
    
    # The line starts after 'CASCI converged'
    pattern = r'CASCI converged'
    # Open the filename within the path
    with open(FilePath) as f:
        # Enumerate gives the index of that line
        for linecounter,line in enumerate(f):
            # When the pattern is found within file
            if re.match(pattern, line):
                # Return index of that line
                return linecounter
            
def ReadSimEnergiesFunc(FilePath):
    
    # The line starts with State'
    pattern = r'^State'
    # Open the filename within the path
    with open(FilePath) as f:
        # Enumerate gives the index of that line
        for linecounter,line in enumerate(f):
            # When the pattern is found within file
            if re.match(pattern, line):
                # Return index of that line
                return linecounter
            
def CASCITheoryData(FilePath, SkipRows):
    """
    Full CI data from output file
    """

    from io import StringIO
    #Stores each line in output file
    with open(FilePath,"r") as f:
        TableLines = f.readlines()
    #Stores the lines after the amount specified by skiprows
    TableLines = TableLines[SkipRows+1:]
    # Filter only desired spin state
    TableLines = [Line for Line in TableLines if "S^2 = 0.0000000" in Line]
    sepPattern = re.compile(
        r"(?P<State>\d{1,3}\s*)"
        r"E\s*=\s+(?P<E>[-]\d[.]\d{14})"
    )
    DesiredData = []
    for Line in TableLines:
        match = re.search(sepPattern,Line)
        if match:
            DesiredData.append({
                "State":int(match.group('State')),
                "E":float(match.group('E'))})
    return DesiredData
    
def CASCISimData(FilePath,SkipRows):
    """
    Simulation data
    """
    from io import StringIO
    #Stores each line in output file
    with open(FilePath,"r") as f:
        TableLines = f.readlines()
    #Stores the lines after the amount specified by skiprows
    TableLines = TableLines[SkipRows+1:]
    # Filter only desired spin state
    sepPattern = re.compile(
        r"\s*(?P<State>\d{1,3}\s*)"
        r"\s+(?P<E>[-]\d[.]\d{8})"
        r"\s+(?P<Exc>\d{1,3}[.]\d{6})"
    )
    SimData = []
    for Line in TableLines:
        match = re.search(sepPattern,Line)
        if match:
            SimData.append({
                "State":int(match.group('State')),
                "Energy (Eh)":float(match.group('E')),
                "Excitation energy (eV)":float(match.group('Exc'))})
    return SimData

def CleanData(DesiredValues: list[dict], SimValues: list[dict]) -> list[dict]:
    desiredEnergy = {"%.8f"%(d['E']) for d in DesiredValues}
    desiredEnergy = {float(i) for i in desiredEnergy}
    cleaned = []
    for energy in SimValues:
        if energy['Energy (Eh)'] in desiredEnergy:
            cleaned.append(energy)
    return cleaned

FilePath = "./test200root.out"
SkipRows = ReadDesiredEnergiesFunc(FilePath)
CASCIDesiredEnergies = CASCITheoryData(FilePath, SkipRows)

SkipRows = ReadSimEnergiesFunc(FilePath)
CASCISimEnergies = CASCISimData(FilePath, SkipRows)

result = CleanData(CASCIDesiredEnergies, CASCISimEnergies)
df = pd.DataFrame(result)
print(df)
#TODO created a filter that Theo wanted to remove the spins that are non-zero


