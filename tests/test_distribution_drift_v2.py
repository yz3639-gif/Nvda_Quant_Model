import numpy as np

from methods.monte_carlo import estimate_gbm_parameters, simulate_gbm_paths


def test_fitted_log_drift_is_not_subtracted_twice():
    returns = np.tile([-.04, .042], 300)
    params = estimate_gbm_parameters(returns)
    paths = simulate_gbm_paths(100, 21, 100000, params.mu_annual, params.sigma_annual, np.random.default_rng(19))
    actual = np.log(paths[:, -1] / 100).mean()
    expected = returns.mean()*21
    se = returns.std(ddof=1)*np.sqrt(21/100000)
    assert abs(actual-expected) < 4*se
