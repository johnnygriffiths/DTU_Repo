import numpy as np

import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# RegEx, creates markers/patterns to get only specific data frfom the files 
import re

def ReadDesiredEnergiesFunc(FilePath: str) -> int:
    
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
            
def ReadSimEnergiesFunc(FilePath: str) -> int:
    
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
            
def CASCITheoryData(FilePath: str, SkipRows: int) -> list[dict]:
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
    # Obtain from each line only the state number and energy (Eh)
    sepPattern = re.compile(
        r"(?P<State>\d{1,3}\s*)"
        r"E\s*=\s+(?P<E>[-]\d[.]\d{14})"
    )
    # Store these in new list[dict]
    DesiredData = []
    for Line in TableLines:
        match = re.search(sepPattern,Line)
        if match:
            DesiredData.append({
                "State":int(match.group('State')),
                "E":float(match.group('E'))})
    return DesiredData
    
def CASCISimData(FilePath: str,SkipRows: int) -> list[dict]:
    """
    Simulation data
    """
    
    #Stores each line in output file
    with open(FilePath,"r") as f:
        TableLines = f.readlines()
    #Stores the lines after the amount specified by skiprows
    TableLines = TableLines[SkipRows+1:]
    # Obtain from each line the state number, the energy (Eh) and the excitation energy (eV)
    sepPattern = re.compile(
        r"\s*(?P<State>\d{1,3}\s*)"
        r"\s+(?P<E>[-]\d[.]\d{8})"
        r"\s+(?P<Exc>\d{1,3}[.]\d{6})"
    )
    # Store these in new list[dict]
    SimData = []
    for Line in TableLines:
        match = re.search(sepPattern,Line)
        if match:
            SimData.append({
                "State":int(match.group('State')),
                "Energy (Eh)":float(match.group('E')),
                "Excitation energy (eV)":float(match.group('Exc'))})
    return SimData

def ComparisonData(DesiredValues: list[dict], SimValues: list[dict]) -> list[dict]:
    """
    Compares the values produced by simulation to CI calculation
    Removes all that are not the same (error from pySCF not maintaining desired spin symmetry)
    """
    desiredEnergy = {"%.8f"%(d['E']) for d in DesiredValues}
    desiredEnergy = {float(i) for i in desiredEnergy}
    FilteredData = []
    for energy in SimValues:
        if energy['Energy (Eh)'] in desiredEnergy:
            FilteredData.append(energy)
    return FilteredData

FilePath = "./test200root.out"
SkipRows = ReadDesiredEnergiesFunc(FilePath)
CASCIDesiredEnergies = CASCITheoryData(FilePath, SkipRows)

SkipRows = ReadSimEnergiesFunc(FilePath)
CASCISimEnergies = CASCISimData(FilePath, SkipRows)

FilteredData = ComparisonData(CASCIDesiredEnergies, CASCISimEnergies)
df = pd.DataFrame(FilteredData)
print(df)