
#!/usr/bin/env python
"""MPI Weight Optimization for EDC Score
Usage: mpiexec -np 8 python mpi_weight_optimization.py
"""
import numpy as np
import pandas as pd
import math
import time
from mpi4py import MPI
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
import pickle
import itertools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Load precomputed data (all ranks) ---
# In practice, master broadcasts data to workers
edc_df = pd.read_csv("PubChem_EDC.csv")
neg_df = pd.read_csv("negative_list.csv")

edc_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) 
              for s in edc_df["SMILES"].dropna() if Chem.MolFromSmiles(str(s))]
neg_smiles = neg_df["smiles"].tolist()

# Precompute fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
edc_fps = [mfpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in edc_smiles[:3000]]

# Precompute scaffold counter (simplified)
from collections import Counter
scaffold_counter = Counter()
for s in edc_smiles[:3000]:
    mol = Chem.MolFromSmiles(s)
    if mol:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            sc_smi = Chem.MolToSmiles(sc)
            if sc_smi:
                scaffold_counter[sc_smi] += 1
        except:
            pass

max_log_freq = math.log(scaffold_counter.most_common(1)[0][1] + 1)

# Property params (precomputed from EDC)
prop_params = {}  # Would load from file in production

def compute_subscores(smi):
    """Compute 4 sub-scores for a SMILES"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # Simplified property score
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    prop_s = math.exp(-0.5*((mw-360)/170)**2) * math.exp(-0.5*((logp-4.6)/3.1)**2)

    # Scaffold score
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        sc_smi = Chem.MolToSmiles(sc)
        count = scaffold_counter.get(sc_smi, 0)
        scaff_s = math.log(count + 1) / max_log_freq if count > 0 else 0.05
    except:
        scaff_s = 0.05

    # Similarity score
    fp = mfpgen.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(fp, edc_fps)
    sim_s = max(sims) if sims else 0.0

    # Fragment score (simplified)
    frag_s = 0.0
    patterns = ["[OX2H]c1ccccc1", "[F,Cl,Br]c", "c1ccc(-c2ccccc2)cc1"]
    matched = sum(1 for p in patterns if mol.HasSubstructMatch(Chem.MolFromSmarts(p)))
    frag_s = matched / len(patterns)

    return (max(prop_s, 1e-10), max(scaff_s, 1e-10), max(sim_s, 1e-10), max(frag_s, 1e-10))

def weighted_geomean(subscores, weights):
    """Weighted geometric mean"""
    log_sum = sum(w * math.log(s) for s, w in zip(subscores, weights))
    return math.exp(log_sum / sum(weights))

# --- Weight Grid ---
weight_candidates = [0.5, 1.0, 1.5, 2.0]
all_weight_combos = list(itertools.product(weight_candidates, repeat=4))

if rank == 0:
    print(f"Total weight combinations: {len(all_weight_combos)}")
    print(f"Workers: {size-1}")
    t0 = time.time()

# Precompute sub-scores for sample (all ranks do this)
sample_pos = edc_smiles[:500]
sample_neg = neg_smiles[:500]

pos_subscores = [compute_subscores(s) for s in sample_pos]
pos_subscores = [x for x in pos_subscores if x is not None]
neg_subscores = [compute_subscores(s) for s in sample_neg]
neg_subscores = [x for x in neg_subscores if x is not None]

# --- Master-Worker weight optimization ---
if rank == 0:
    # Distribute weight combos to workers
    results = []
    task_queue = list(range(len(all_weight_combos)))
    finished = 0

    # Initial task distribution
    for worker in range(1, min(size, len(task_queue)+1)):
        if task_queue:
            idx = task_queue.pop(0)
            comm.send(idx, dest=worker, tag=1)

    # Receive results and send new tasks
    while finished < size - 1:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker = status.Get_source()

        if result is not None:
            results.append(result)

        if task_queue:
            idx = task_queue.pop(0)
            comm.send(idx, dest=worker, tag=1)
        else:
            comm.send(-1, dest=worker, tag=1)  # DONE signal
            finished += 1

    # Find best weights
    best = max(results, key=lambda x: x[1])
    print(f"
Best weights: {best[0]}, ROC-AUC: {best[1]:.4f}")
    print(f"Total time: {time.time()-t0:.1f}s with {size} processes")

    # Save results
    pd.DataFrame(results, columns=["weights", "auc"]).to_csv("weight_optimization_results.csv", index=False)

else:
    # Worker loop
    while True:
        idx = comm.recv(source=0, tag=1)
        if idx == -1:
            break

        weights = all_weight_combos[idx]

        # Compute AUC for this weight combo
        pos_final = [weighted_geomean(ss, weights) for ss in pos_subscores]
        neg_final = [weighted_geomean(ss, weights) for ss in neg_subscores]

        y_true = np.concatenate([np.ones(len(pos_final)), np.zeros(len(neg_final))])
        y_score = np.concatenate([pos_final, neg_final])

        try:
            auc_val = roc_auc_score(y_true, y_score)
        except:
            auc_val = 0.5

        comm.send((weights, auc_val), dest=0, tag=2)
