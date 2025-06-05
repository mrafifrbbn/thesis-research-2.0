Experiment to set different logdist priors. Mainly testing the prior P(r) ~ r^2, which corresponds to P(eta) = 10^(-3 * eta)

1. Removing 2pi in the posterior distribution calculation -> as expected, doesn't change the logdists
2. Adding constants (but different value for each data point) ->
Also tested how adding varying constants change the posterior distribution. Shouldn't change -> 