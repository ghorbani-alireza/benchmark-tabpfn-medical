import numpy as np
from scipy.linalg import block_diag


def _make_settings():
    S = {}

    # ── Setting 1 ─────────────────────────────────────────
    S[1] = dict(
        label='Setting 1: Small Sample, Low-Dimensional Gaussian',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(20), 'mu2': np.array([0.7]*10 + [0]*10)},
        sigma1=np.eye(20),
        sigma2=np.diag([1.3]*10 + [1]*10),
        dist='Normal'
    )

    # ── Shared covariance for settings 2 & 3 ─────────────
    block1 = np.full((10, 10), 0.4) + np.eye(10) * (1 - 0.4)
    block2 = np.eye(10)
    sig23_1 = block_diag(block1, block2)
    sig23_2 = np.linalg.inv(np.linalg.inv(sig23_1) + np.eye(20))

    # ── Setting 2 ─────────────────────────────────────────
    S[2] = dict(
        label='Setting 2: Nonlinear Interactions with No Mean Shiftyou fucking kidding me wh',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(20), 'mu2': np.zeros(20)},
        sigma1=sig23_1, sigma2=sig23_2,
        dist='Normal'
    )

    # ── Setting 3 ─────────────────────────────────────────
    S[3] = dict(
        label='Setting 3: Nonlinear Interactions with Sparse Linear Signal',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(20),
            'mu2': np.concatenate([np.ones(5), np.zeros(15)])},
        sigma1=sig23_1, sigma2=sig23_2,
        dist='Normal'
    )

    # ── Setting 4 ─────────────────────────────────────────
    S[4] = dict(
        label='Setting 4: Class Imbalance',
        n={'n1_tr': 160, 'n2_tr': 40, 'n1_te': 800, 'n2_te': 200},
        mu={'mu1': np.zeros(20), 'mu2': np.array([0.7]*10 + [0]*10)},
        sigma1=np.eye(20),
        sigma2=np.diag([1.3]*10 + [1]*10),
        dist='Normal'
    )

    # ── Setting 5 ─────────────────────────────────────────
    S[5] = dict(
        label='Setting 5: Heavy-Tailed Elliptical Distributions',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(20), 'mu2': np.array([0.7]*10 + [0]*10)},
        sigma1=np.eye(20),
        sigma2=np.diag([1.3]*10 + [1]*10),
        dist='t', df=5,
    )

    # ── Setting 6 ─────────────────────────────────────────
    sig6_2 = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            sig6_2[i, j] = 0.6 ** abs(i - j)
    S[6] = dict(
        label='Setting 6: Heavy-Tailed with Exponential Covariance Decay',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(20), 'mu2': np.array([0.7]*10 + [0]*10)},
        sigma1=np.eye(20), sigma2=sig6_2,
        dist='t', df=5,
    )

    # ── Setting 7 ─────────────────────────────────────────
    S[7] = dict(
        label='Setting 7: Sparse Gaussian (p=40)',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(40), 'mu2': np.array([0.7]*5 + [0]*35)},
        sigma1=np.eye(40),
        sigma2=np.diag([1.3]*5 + [1]*35),
        dist='Normal'
    )

    # ── Setting 8 ─────────────────────────────────────────
    S[8] = dict(
        label='Setting 8: High Dimensional Sparse Gaussian (p=100)',
        n={'n1_tr': 100, 'n2_tr': 100, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(100), 'mu2': np.array([0.7]*5 + [0]*95)},
        sigma1=np.eye(100),
        sigma2=np.diag([1.3]*5 + [1]*95),
        dist='Normal'
    )

    return S


SETTINGS = _make_settings()


def _make_sensitivity():
    """Same settings, larger training samples (n_tr=500 or 500/100)."""
    S = {}
    for sid, cfg in SETTINGS.items():
        new = dict(cfg)
        if sid == 4:
            new['n'] = {'n1_tr': 400, 'n2_tr': 100, 'n1_te': 800, 'n2_te': 200}
        else:
            new['n'] = {'n1_tr': 250, 'n2_tr': 250, 'n1_te': 500, 'n2_te': 500}
        new['label'] = cfg['label'] + ' (Sensitivity)'
        S[sid] = new

    # Setting 7+ : 40-dim with block covariance + t-dist
    block1 = np.full((20, 20), 0.4) + np.eye(20) * (1 - 0.4)
    block2 = np.eye(20)
    sig_7p = block_diag(block1, block2)
    S['7+'] = dict(
        label='Setting 7+ (Sensitivity, t-dist, block covariance)',
        n={'n1_tr': 250, 'n2_tr': 250, 'n1_te': 500, 'n2_te': 500},
        mu={'mu1': np.zeros(40), 'mu2': np.array([0.7]*5 + [0]*35)},
        sigma1=sig_7p,
        sigma2=np.linalg.inv(np.linalg.inv(sig_7p) + np.eye(40)),
        dist='t', df=5,
    )
    return S


SENSITIVITY = _make_sensitivity()