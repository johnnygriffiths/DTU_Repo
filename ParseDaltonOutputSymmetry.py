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
    """
    Oscillator strength only from ungerade symmetries:
    Symmetry 2 (B3u), 3 (B2u), 5 (B1u)
    as this is symmetry of the dipole stretching
    """
    OscStrData = []
    for OscStrLine in OscStrLines:
        match = re.search(OscStrPattern, OscStrLine)
        if match:
            OscStrData.append(float(match.group('OscStr')) #if want specific d.p then ('%f%' % float(..))
            )
    #Store in dataframe
    EnergDataDF = pd.DataFrame({"Excitation energy (eV)": EnergData}) #,"Total oscillator strength": OscStrData})
    OscStrDataDF = pd.DataFrame({"Total oscillator strength": OscStrData})

    # Join dataframes together by columns
    df = pd.concat([EnergDataDF,OscStrDataDF], axis=1)
    # Molecular symmetry in order of dalton output
    sym_names = ["Ag","B3u","B2u","B1g","B1u","B2g","B3g","Au"]
    name_map = dict(zip(range(1,9),sym_names))

    # Where the current value in the excitation energy is smaller than the previous
    # must be the next symmetry group --> increment symmetry counter by 1
    # df.diff() gets the difference between current row element and previous row element
    # < 0 turns the diff into boolean 
    # cumsum() treats bool as 0 and 1, so just increments from 1 to 8
    df['Symmetry'] = (df['Excitation energy (eV)'].diff() < 0).cumsum() + 1
    # Where row has 1, 2, etc., map it to element in name_map dict
    df['Symmetry'] = df['Symmetry'].map(name_map)

    # Number of variables of each symmetry
    # For adding the same number of oscillator strengths to the correct symmetry
    sym_sizes = df.groupby('Symmetry').size()

    # Which symmetry groups have oscillator strengths
    # only ungerade symmetry groups
    ungerade_sym = [symmetry for symmetry in sym_names if "u" in symmetry] 
    # A1u is 1-dimensional and doesn't have oscillator strength 
    ungerade_sym.remove("Au")

    # Set elements to NaN where the symmetry is gerade
    df['Total oscillator strength'] = np.nan
    symmetries = OscStrDataDF['Total oscillator strength'].tolist()
    pointer = 0

    # for each symmetry in the symmetries with oscillator strength
    for sym in ungerade_sym:
        # get the number of variables in that symmetry
        sym_variables = sym_sizes[sym]
        # to assign the same amount of oscillator strengths
        chunk = symmetries[pointer : pointer + sym_variables]
        pointer += sym_variables

        # gets index of lines with desired symmetry
        idx = df.index[df['Symmetry'] == sym]
        # gets the index and matches them to oscillator strength values
        df.loc[idx, 'Total oscillator strength'] = chunk
    
    return df

FileName = "CAS67_sym_beh2_sym_sto3g"
FilePath = f"./DaltonOutputs/{FileName}.out"

CASCISimEnergiesDF = CASCISimData(FilePath)
# The index count is the same as the root set in the Dalton file, i.e the number of excited states
CASCISimEnergiesDF.index +=1
CASCISimEnergiesDF = CASCISimEnergiesDF.drop(columns=['Symmetry'])
CASCISimEnergiesDF = CASCISimEnergiesDF.sort_values(by=["Excitation energy (eV)"])
print(CASCISimEnergiesDF)
CASCISimEnergiesDF.to_latex(f"{FileName}123123.tex", index=False, longtable=True)