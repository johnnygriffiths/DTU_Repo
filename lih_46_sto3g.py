import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from pyscf.fci import addons

# Build molecule (use a reasonable bond length, e.g. 1.6 Å)
mol = gto.M(
    atom = 'Li 0 0 0; H 0 0 1.595',
    basis = 'sto-3g',
    spin = 0,   # singlet reference
    verbose=4,
    charge = 0
)

# RHF calculation
mf = scf.RHF(mol)
mf.kernel()

# CASCI: 4 electrons in 6 orbitals
# select active pace (li has 4e), then selects certain number orbitals, some ooccu, some virtual
# then within this set it does FCI (here sto-3g does fci anyway)
ncas = 6
nelecas = 4

mc = mcscf.CASCI(mf, ncas, nelecas) 

# Request multiple states (e.g. first 10 singlets)
# 105 is maximum number of states inclusive ground state
nroots = 400

# Enforce singlet states using spin-adapted FCI solver
#solver = fci.direct_spin1.FCI(mol)
solver = fci.direct_spin0.FCI(mol)
#solver = fci.addons.fix_spin_(solver, ss=0.0, shift=2.0)
solver.spin = 0  # target S=0
solver.nroots = nroots

mc.fcisolver = solver

# Run CASCI
mc.kernel()

# Extract energies
energies = mc.e_tot  # ground and excited states

# Convert to excitation energies (relative to ground state)
E0 = energies[0]
excitation_au = energies - E0

# Convert to eV
hartree_to_ev = 27.211386246
excitation_ev = excitation_au * hartree_to_ev

print("State   Energy (Ha)   Excitation (eV)")
for i, (E, dE) in enumerate(zip(energies, excitation_ev)):
    print(f"{i:2d}   {E:12.8f}   {dE:10.6f}")



# AO dipole integrals (x,y,z)
mu_ao = mol.intor('int1e_r', comp=3)

# transform to MO basis
mo = mc.mo_coeff
mu_mo = np.einsum('xij,ip,jq->xpq', mu_ao, mo, mo)


E = mc.e_tot
E0 = E[0]


for i in range(1, len(E)):
    dE = E[i] - E0

    ci0 = mc.ci[0]
    cii = mc.ci[i]

    tdm1 = mc.fcisolver.trans_rdm1(ci0, cii, mc.ncas, mc.nelecas)

    mu_vec = np.einsum('xij,ij->x', mu_mo, tdm1)

    f = (2.0/3.0) * dE * np.dot(mu_vec, mu_vec)

    print(f"State {i}: dE = {dE*hartree_to_ev:.6f} eV, f = {f:.6f}")


