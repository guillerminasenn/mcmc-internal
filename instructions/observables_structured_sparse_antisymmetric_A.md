# MCMC Observables for Structured Sparse Antisymmetric Matrix A

We consider a matrix $A \in \mathbb{R}^{d \times d}$ such that:
- $A^\top = -A$ (antisymmetric)
- $a_{ii} = 0$
- We work with the upper triangle $i < j$

## 1. Band-Limited Energy

### Formula
$$
E_k(A) = \sum_{\substack{i<j \\ |i-j| \leq k}} a_{ij}^2
$$

### Explanation
This observable measures the energy of interactions within a band of width $k$ around the diagonal. It aligns with priors that penalize long-range interactions and avoids dilution from far-off coefficients that are near zero.

### Implementation notes
- Loop only over $i < j$
- Apply condition $|i - j| \le k$

## 2. Row Norms

### Formula (row $i$)
$$
R_i(A) = \sum_{j > i} a_{ij}^2
$$

### Alternative (full row norm)
$$
R_i^{\text{full}}(A) = \sum_{j \ne i} a_{ij}^2
$$

### Explanation
Measures the strength of interactions associated with a given row. Unlike means, norms do not vanish under sparsity and capture local structure.

### Implementation notes
- Use upper triangle for consistency, or full row if preferred

## 3. Maximum Entry (Order Statistic)

### Formula
$$
M(A) = \max_{i<j} |a_{ij}|
$$

### Explanation
Captures the largest active interaction in the system. Robust to sparsity and useful to diagnose whether the chain explores high-amplitude regions.

## 4. Top-k Average (Order Statistics)

### Formula
Let $\{|a_{ij}|\}_{i<j}$ sorted in decreasing order:
$$
T_k(A) = \frac{1}{k} \sum_{\ell=1}^{k} |a|_{(\ell)}
$$

### Explanation
Average of the $k$ largest absolute entries. More stable than the maximum and still insensitive to the bulk of near-zero coefficients.

### Implementation notes
- Flatten upper triangle
- Take absolute values
- Sort descending (or use partial sort)
- Average top $k$

## 5. Leading Singular Value (Spectral Norm)

### Formula
$$
\sigma_{\max}(A) = \|A\|_2
$$

### Explanation
Measures the strongest linear action of the operator. For antisymmetric matrices, singular values correspond to magnitudes of purely imaginary eigenvalues, making this a natural measure of transport strength.

### Implementation notes
- Use SVD or power iteration for large $d$

## 6. Support Size (Thresholded Sparsity)

### Formula
$$
S_\varepsilon(A) = \#\{(i,j) : i < j,\ |a_{ij}| > \varepsilon\}
$$

### Explanation
Counts the number of active coefficients above a threshold $\varepsilon$. Provides a proxy for sparsity and helps detect whether the chain is stuck in overly sparse regions.

### Implementation notes
- Choose $\varepsilon$ relative to noise level or prior scale
- Optionally normalize by total number of entries

## Practical Tips

- Always operate on the upper triangle to avoid redundancy
- Precompute index masks for:
  - upper triangle
  - band structure
- For large $d$, use partial sorting (e.g., `np.partition`) instead of full sort
