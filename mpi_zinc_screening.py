
#!/usr/bin/env python
"""MPI Large-Scale ZINC Screening
Usage: mpiexec -np 16 python mpi_zinc_screening.py
"""
import numpy as np
import pandas as pd
import math, time, glob, os
from mpi4py import MPI
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Precompute EDC reference data (all ranks)
edc_df = pd.read_csv("PubChem_EDC.csv")
edc_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
              for s in edc_df["SMILES"].dropna()[:3000] if Chem.MolFromSmiles(str(s))]
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
edc_fps = [mfpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in edc_smiles]

# Scaffold counter
scaffold_counter = Counter()
for s in edc_smiles:
    mol = Chem.MolFromSmiles(s)
    if mol:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            sc_smi = Chem.MolToSmiles(sc)
            if sc_smi: scaffold_counter[sc_smi] += 1
        except: pass
max_log_freq = math.log(scaffold_counter.most_common(1)[0][1] + 1)

# Optimized weights (from weight optimization)
WEIGHTS = (1.0, 1.0, 1.5, 1.0)  # property, scaffold, similarity, fragment

def score_molecule(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    prop_s = max(math.exp(-0.5*((mw-360)/170)**2) * math.exp(-0.5*((logp-4.6)/3.1)**2), 1e-10)
    try:
        sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        scaff_s = max(math.log(scaffold_counter.get(sc, 0) + 1) / max_log_freq, 0.05)
    except: scaff_s = 0.05
    fp = mfpgen.GetFingerprint(mol)
    sim_s = max(max(DataStructs.BulkTanimotoSimilarity(fp, edc_fps)), 1e-10)
    patterns = ["[OX2H]c1ccccc1", "[F,Cl,Br]c", "c1ccc(-c2ccccc2)cc1"]
    matched = sum(1 for p in patterns if mol.HasSubstructMatch(Chem.MolFromSmarts(p)))
    frag_s = max(matched / len(patterns), 1e-10)
    log_sum = sum(WEIGHTS[i]*math.log(s) for i, s in enumerate([prop_s, scaff_s, sim_s, frag_s]))
    return math.exp(log_sum / sum(WEIGHTS))

# Master-Worker dynamic task queue
zinc_files = sorted(glob.glob("../zinc_db/*.txt"))

if rank == 0:
    t0 = time.time()
    task_queue = list(range(len(zinc_files)))
    all_hits = []
    finished = 0

    for worker in range(1, min(size, len(task_queue)+1)):
        if task_queue:
            comm.send(task_queue.pop(0), dest=worker, tag=1)

    while finished < size - 1:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker = status.Get_source()
        if result:
            all_hits.extend(result)
        if task_queue:
            comm.send(task_queue.pop(0), dest=worker, tag=1)
        else:
            comm.send(-1, dest=worker, tag=1)
            finished += 1

    elapsed = time.time() - t0
    print(f"Screening done: {len(all_hits)} hits (score>0.3) in {elapsed:.1f}s with {size} procs")
    pd.DataFrame(all_hits, columns=["smiles","score"]).to_csv("zinc_screening_hits.csv", index=False)

else:
    while True:
        idx = comm.recv(source=0, tag=1)
        if idx == -1: break
        hits = []
        try:
            df = pd.read_csv(zinc_files[idx], sep="\t", usecols=["smiles"])
            for smi in df["smiles"]:
                s = score_molecule(smi)
                if s and s > 0.3:
                    hits.append((smi, s))
        except: pass
        comm.send(hits, dest=0, tag=2)
