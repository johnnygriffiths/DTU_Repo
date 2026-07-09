"""
Stretching experiment of BeH2 - Parallelized Version
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from pySCFsims.casci_pyscf import (
    MoleculeData,
    DMDMWorkflow,
)
#from dmdm.interface import DMDM
#import qrunch as qc

def run_single_factor(args):
    """Worker function for each factor - must be at module level for pickling"""
    factor, num_active_orbitals, num_active_electrons, num_states = args
    
    print(f"Processing factor: {factor}")
    
    # # Create calculator fresh in each worker (avoid pickling issues)
    # calculator = (qc.calculator_creator()
    #                 .vqe()
    #                 .iterative()
    #                 .standard()
    #                 .with_options(
    #                         options=qc.options.IterativeVqeOptions(max_iterations=500)
    #                 )
    #                 .create()
    # )
    calculator = None
    calculator2 = None
    # calculator2= (qc.calculator_creator()
    #                 .vqe()
    #                 .iterative()
    #                 .standard()
    #                 .with_options(
    #                         options=qc.options.IterativeVqeOptions(max_iterations=500)
    #                 )
    #                 .create()
    # )
    
    mol = [
            ("Be", 0.0, 0.0, 0.0),
            ("H", 0.0, 0.0, factor),
            ("H", 0.0, 0.0,-factor),
        ]
    
    workflow = DMDMWorkflow(
        basis="cc-pvdz",
        molecule=mol,
        num_active_orbitals=num_active_orbitals,
        num_active_electrons=num_active_electrons,
        num_states=num_states,
        calculator=calculator,
        casci_like=True
    )

    workflow2 = DMDMWorkflow(
        basis="cc-pvdz",
        molecule=mol,
        num_active_orbitals=num_active_orbitals,
        num_active_electrons=num_active_electrons,
        num_states=num_states,
        calculator=calculator2,
        casci_like=True
    )
    
    # Run all calculations
    # result_vqe = workflow.run_quantum_vqe()
    # result_vqe_oo = workflow2.run_quantum_vqe()
    # result_casci_dmdm = workflow.run_classical_casci_dmdm()
    result_casci = workflow.run_classical_casci()
    
    return {
        'factor': factor,
        # 'vqe': {
        #     "state": [*range(1, len(result_vqe["exc_energies_ev"])+1)],
        #     "energy_ev": list(result_vqe["exc_energies_ev"]),
        #     "oscillator_strength": list(result_vqe["oscillator_strengths"])
        # },
        # 'vqe_oo': {
        #     "state": [*range(1, len(result_vqe_oo["exc_energies_ev"])+1)],
        #     "energy_ev": list(result_vqe_oo["exc_energies_ev"]),
        #     "oscillator_strength": list(result_vqe_oo["oscillator_strengths"])
        # },
        'casci': {
            "state": [*range(1, len(result_casci["exc_energies_ev"])+1)],
            "energy_ev": list(result_casci["exc_energies_ev"]),
            "oscillator_strength": list(result_casci["oscillator_strengths"])
        },
        # 'casci_dmdm': {
        #     "state": [*range(1, len(result_casci_dmdm["exc_energies_ev"])+1)],
        #     "energy_ev": result_casci_dmdm["exc_energies_ev"],
        #     "oscillator_strength": list(result_casci_dmdm["oscillator_strengths"])
        # }
    }


if __name__ == "__main__":
    # Define parameters
    factors = (
        np.round(np.linspace(0.5, 1.3, 9), 2).tolist() + 
        [1.330] + 
        np.round(np.linspace(1.4, 4.0, 27), 2).tolist()
    )
    num_workers = min(cpu_count(), 24)  # Limit workers to avoid memory issues
    
    # Prepare arguments for parallel execution
    args_list = [
        (factor, 7, 6, 1150)  # num_active_orbitals, num_active_electrons, num_states
        for factor in factors
    ]
    
    # Execute in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_factor, args_list)
    
    # Consolidate results (same structure as original)
    vqe_dfs = []
    vqe_oo_dfs = []
    casci_dfs = []
    casci_dmdm_dfs = []
    
    for result in results:
        factor = result['factor']
        
        vqe_dfs.append(pd.DataFrame({
            "state": result['vqe']['state'],
            "energy_ev": result['vqe']['energy_ev'],
            "stretch": [factor] * len(result['vqe']['state']),
            "oscillator_strength": result['vqe']['oscillator_strength']
        }))

        vqe_dfs.append(pd.DataFrame({
            "state": result['vqe_oo']['state'],
            "energy_ev": result['vqe_oo']['energy_ev'],
            "stretch": [factor] * len(result['vqe_oo']['state']),
            "oscillator_strength": result['vqe_oo']['oscillator_strength']
        }))
        
        casci_dfs.append(pd.DataFrame({
            "state": result['casci']['state'],
            "energy_ev": result['casci']['energy_ev'],
            "stretch": [factor] * len(result['casci']['state']),
            "oscillator_strength": result['casci']['oscillator_strength']
        }))
        
        casci_dmdm_dfs.append(pd.DataFrame({
            "state": result['casci_dmdm']['state'],
            "energy_ev": result['casci_dmdm']['energy_ev'],
            "stretch": [factor] * len(result['casci_dmdm']['state']),
            "oscillator_strength": result['casci_dmdm']['oscillator_strength']
        }))
    
    # Save results
    os.makedirs("beh2_stretching", exist_ok=True)
    
    #vqe_df = pd.concat(vqe_dfs, axis=0).reset_index(drop=True)
    #vqe_oo_df = pd.concat(vqe_dfs, axis=0).reset_index(drop=True)
    casci_df = pd.concat(casci_dfs, axis=0).reset_index(drop=True)
    #casci_dmdm_df = pd.concat(casci_dmdm_dfs, axis=0).reset_index(drop=True)
    
    #vqe_df.to_csv("beh2_stretching/vqe_beh2_stretching.csv")
    #vqe_oo_df.to_csv("beh2_stretching/vqe_beh2_stretching.csv")
    casci_df.to_csv("./casci_beh2_stretching.csv")
    #casci_dmdm_df.to_csv("beh2_stretching/casci_dmdm_beh2_stretching.csv")