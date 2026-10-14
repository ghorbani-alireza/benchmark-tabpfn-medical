import numpy as np
from scipy.linalg import block_diag


def _make_settings():
    """Define the eight baseline simulations: sample sizes, class means, covariance matrices, and distributions."""
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
    """Define the larger-training-sample versions of the baseline settings, plus the extra setting 7+."""
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

def calculate_bayes_auc(setting, n_pairs=100_000, seed=314159):
    """Estimate the best population AUC using the known class distributions.

    Class 1 (mu1) is the negative class; class 2 (mu2) is positive, as in
    simulation_core. Score observations by log f_positive - log f_negative.
    AUC is the probability that an independent positive has the higher score,
    counting ties as half. No model is fitted and no benchmark data are used.

    Unequal covariances and t distributions use n_pairs independent pairs.
    The returned mc_se measures numerical Monte Carlo uncertainty, not the
    variation between fitted models. A local seed leaves simulation RNG alone.
    Equal-covariance Gaussian AUC has a closed form, used without simulation.
    """
    from scipy.stats import multivariate_normal, multivariate_t, norm
    from threadpoolctl import threadpool_limits

    if not isinstance(n_pairs, (int, np.integer)) or n_pairs < 2:
        raise ValueError("n_pairs must be an integer of at least 2")
    means = [np.asarray(setting['mu'][key], dtype=float) for key in ('mu1', 'mu2')]
    covs = [np.asarray(setting[key], dtype=float) for key in ('sigma1', 'sigma2')]
    dist = setting['dist']
    if dist not in ('Normal', 't'):
        raise ValueError("Bayes AUC supports only Normal and t distributions")
    # The exact normal formula also gives 0.5 for identical classes.
    if dist == 'Normal' and np.array_equal(covs[0], covs[1]):
        delta = means[1] - means[0]
        distance_squared = delta @ np.linalg.solve(covs[0], delta)
        return dict(auc=float(norm.cdf(np.sqrt(distance_squared / 2))),
                    mc_se=0.0, n_pairs=0, seed=None, method='exact Gaussian')

    if dist == 't':
        df = setting.get('df')
        if df is None or df <= 2:
            raise ValueError("t settings need df > 2 because sigma is a covariance")
        # scipy's t shape is not its covariance: match simulation_core exactly.
        distributions = [multivariate_t(loc=m, shape=c * (df - 2) / df, df=df)
                         for m, c in zip(means, covs)]
    else:
        distributions = [multivariate_normal(mean=m, cov=c) for m, c in zip(means, covs)]
    rng = np.random.default_rng(seed)
    total = total_squared = 0.0
    # Small batches keep memory use bounded, even for the 100-feature setting.
    with threadpool_limits(limits=1):
        for start in range(0, n_pairs, 2_000):
            size = min(2_000, n_pairs - start)
            negative, positive = [d.rvs(size=size, random_state=rng).reshape(size, -1)
                                  for d in distributions]
            negative_score = distributions[1].logpdf(negative) - distributions[0].logpdf(negative)
            positive_score = distributions[1].logpdf(positive) - distributions[0].logpdf(positive)
            wins = (positive_score > negative_score).astype(float)
            wins += 0.5 * (positive_score == negative_score)
            total += wins.sum()
            total_squared += wins @ wins
    auc = total / n_pairs
    variance = max(0.0, (total_squared - n_pairs * auc**2) / (n_pairs - 1))
    return dict(auc=float(auc), mc_se=float(np.sqrt(variance / n_pairs)),
                n_pairs=int(n_pairs), seed=int(seed), method='Monte Carlo likelihood ratio')
