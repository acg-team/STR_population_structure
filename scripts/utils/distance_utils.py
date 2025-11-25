# utils/distance_utils.py

import numpy as np
from scipy.spatial.distance import squareform

def mantel_test(D1, D2, perms=10000, random_state=None):
    """
    Mantel test comparing two distance matrices (Pearson correlation).
    """
    rng = np.random.default_rng(random_state)

    D1 = np.asarray(D1)
    D2 = np.asarray(D2)

    v1 = squareform(D1, checks=False)
    v2 = squareform(D2, checks=False)

    r_obs = np.corrcoef(v1, v2)[0, 1]

    count = 0
    n = D1.shape[0]

    for _ in range(perms):
        perm = rng.permutation(n)
        D2_perm = D2[perm][:, perm]
        v2_perm = squareform(D2_perm, checks=False)
        r_perm = np.corrcoef(v1, v2_perm)[0, 1]
        if r_perm >= r_obs:
            count += 1

    p_value = (count + 1) / (perms + 1)
    return float(r_obs), float(p_value)
