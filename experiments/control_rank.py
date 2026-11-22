"""Fit-free check of the modal-bandwidth (degree-of-freedom) length bound.

Count the generator directions that the programmable index controls at first
order: eigenvalues of the Gram matrix G(L) (pwg.controllability) above 10 % of
the mean diagonal.  Lloyd-Montangero counting predicts
    rank ~ min(N^2, (2N) * (dbeta_max * L / pi)),
i.e. rank / N^2 is a function of L / L_info with L_info = pi N / dbeta_max
~ N n lambda / NA^2, independent of N.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pwg.controllability import Controllability
from pwg.mmwg import MMWGConfig, MultimodeWaveguide

out = []
for na in [0.3, 0.2, 0.45]:
    for N in [8, 16, 32, 64]:
        wg = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=100.0, na=na))
        C = Controllability(wg)
        dbmax = float(wg.beta.max() - wg.beta.min())
        L_info = math.pi * N / dbmax
        for f in np.logspace(-1.5, 0.7, 14):
            L = f * L_info
            H, _ = C.h(L, 1.0)
            G = C.X[np.ix_(C.row_pair, C.row_pair)] * (H @ H.T)
            ev = np.linalg.eigvalsh(G)
            rank = int((ev > 0.1 * np.mean(np.diag(G))).sum())
            out.append(dict(N=N, na=na, L_um=L, L_info=L_info, dbeta_max=dbmax, rank=rank, R=N * N))
            print(json.dumps(out[-1]), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "control_rank.json"), "w"), indent=1)
