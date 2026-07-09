# Diagnostics

## RMSE/MAE/MSE diagnostics (solute transport)

For the solute transport convergence notebook RMSE section:

1. Choose component index i for the parameter vector and the max iteration counts for the bar and time-series diagnostics.
2. Compute per-chain MAE, MSE, and RMSE using the first n samples:
   - MAE = mean(abs(x_i - true_i))
   - MSE = mean((x_i - true_i)^2)
   - RMSE = sqrt(MSE)
3. For thinned pCN, thin each pCN chain by stride P, then limit the unthinned iterations so that unthinned length equals P times the thinned length used.
4. Bar plot: compute RMSE per chain (using the chosen iteration counts), then take the mean over chains for each algorithm (mPCN, pCN, pCN thinned).
5. Time-series plots: for k = 1..K, compute MAE/MSE/RMSE using the first k samples of each chain; average over chains within each algorithm to get the plotted series.

### Running MSE: calculation details

Use the following observable IDs for the running MSE plots:
1. $\bar{x}_{1:d-1}$: mean of the first $d-1$ components.
2. $\mathrm{Var}(x_{1:d-1})$: variance of the first $d-1$ components.
3. $x_0$: first component.
4. $\min_i x_i$: minimum over all components.
5. $\bar{x}=\frac{1}{d}\sum_i x_i$: mean over all components.
6. $\mathrm{Var}(x)$: variance over all components.
7. $\|x\|_2$: Euclidean norm.
8. $\Phi(x)=\frac{1}{2\sigma^2}\|y-f(x)\|^2$: potential.

For a chain $\{X_t\}_{t=1}^T$ and observable $\varphi$, the running mean is
$$\bar{\varphi}_t=\frac{1}{t}\sum_{s=1}^t \varphi(X_s).$$
For EP with $P$ parallel chains, first average per iteration
$$\tilde{\varphi}_t^{\mathrm{EP}}=\frac{1}{P}\sum_{p=1}^P \varphi(X_{t,p}),$$
then compute the running mean over iterations
$$\bar{\varphi}_t^{\mathrm{EP}}=\frac{1}{t}\sum_{s=1}^t \tilde{\varphi}_s^{\mathrm{EP}}.$$
This matches averaging per-chain running means at each $t$.

Reference targets are posterior means estimated from long mPCN chains after burn-in $10{,}000$:
$$\mu_\varphi=\frac{1}{K}\sum_{k=1}^K \Big(\frac{1}{T-10{,}000}\sum_{t=10{,}001}^T \varphi(X_t^{(k)})\Big).$$

At each iteration $t$, the running MSE is averaged across chains (or EP groups):
$$\mathrm{MSE}_t=\frac{1}{K}\sum_{k=1}^K\big(\bar{\varphi}_t^{(k)}-\mu_\varphi\big)^2.$$

Plot selection is controlled by the `mse_observable_ids` list in the running-MSE section; defaults omit IDs 5 and 6 (mean and variance over all components).

## Independent pCN ESS/MSJD computation

When plotting or tabulating independent pCN diagnostics for a given rho and P:

1. Load the P independent pCN chain files.
2. Thin each chain by taking every P-th sample (stride = P), then apply burn-in.
3. Compute ESS per chain on the thinned samples and sum across chains.
4. Compute MSJD per chain on the thinned samples, then aggregate:
   - unweighted mean of the per-chain MSJD means

These aggregations are stored in the independent metrics payload:
- ess_mean_sum
- msjd_mean_mean
- msjd_mean_max (only if explicitly computed in a notebook)
