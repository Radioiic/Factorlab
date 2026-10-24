import numpy as np

from factorlab import stats as S


def test_bh_qvalues_reference():
    p = np.array([0.01, 0.04, 0.03, 0.5, 0.001])
    # reference: q_(i) = min_{j>=i} p_(j) * n / j
    order = np.argsort(p)
    n = p.size
    ref = np.empty(n)
    ps = p[order]
    for i in range(n):
        ref[order[i]] = min(min(ps[j] * n / (j + 1) for j in range(i, n)), 1)
    assert np.allclose(S.bh_qvalues(p), ref)
