# %%
import time
import torch as pt  

# pt.backends.cuda.preferred_linalg_library('magma')  # or 'cusolver'

# %%
batch_size = 1
d = 4000
t = 10000

# t samples in d dimensions, living in a rank-dimensional subspace + noise
rank = 400
components = pt.randn(batch_size, rank, d, device="cuda")       # rank basis vectors in R^d
coeffs = pt.randn(batch_size, t, rank, device="cuda")            # t samples as linear combos
Data = coeffs @ components + 0.1 * pt.randn(batch_size, t, d, device="cuda")
print(f"Data.shape: {Data.shape}")

# %%
cov = (Data.mT @ Data)
G = cov
true_U, true_S, true_V = pt.linalg.svd(G)

def ev(num):
    pt.cuda.synchronize()
    print(f"post {num}: {time.time() - start_time:.3f} seconds")

# %%

q = 400
niter = 0

start_time = time.time()


# U, S, V = pt.svd_lowrank(G, q=400, niter=1)


# Step 0: compute covariance matrix
G = (Data.mT @ Data)
ev(0)

# Step 1: random projection  A @ Omega  where Omega is (n, q)
Omega = pt.randn(G.shape[-1], q, dtype=G.dtype, device=G.device)
Y = G @ Omega
ev(1)

# Step 2: QR to get orthonormal basis for the column space
Q = pt.linalg.qr(Y).Q
ev(2)

# Step 3: power iterations to sharpen the spectrum
for _ in range(niter):
    Z = pt.linalg.qr(G.transpose(-1, -2) @ Q).Q
    Q = pt.linalg.qr(G @ Z).Q
ev(3)

# Step 4: project to low-rank: B = Q^T A  (small q x n matrix)
B = Q.transpose(-1, -2) @ G
ev(4)

# Step 5: exact SVD of the small matrix B
Ub, S, V = pt.linalg.svd(B, full_matrices=False)
V = V.transpose(-1, -2)
ev(5)

# Step 6: recover full-size left singular vectors
U = Q @ Ub
ev(6)

# relative reconstruction error: ||G - U S V^T|| / ||G||
approx = U * S.unsqueeze(-2) @ V.transpose(-1, -2)
rel_error = pt.linalg.norm(G - approx, dim=(-2, -1)) / pt.linalg.norm(G, dim=(-2, -1))
print(f"Relative reconstruction error: {rel_error.mean():.6f}")

# %% cov in small dim space

q = 400

start_time = time.time()

P = pt.randn(Data.shape[-1], q, dtype=Data.dtype, device=Data.device)
P = pt.linalg.qr(P).Q
U_full = P
ev(0)


# power iteration: P_new = Data^T (Data @ P_old_eigvecs) = C @ U_full
# but we compute it without forming C, via two passes through Data
P = Data.mT @ (Data @ U_full)  # d × q  (= C @ U_full)
P = pt.linalg.qr(P).Q
ev("iter1")



# project data and form small covariance: P^T (X^T X) P
Y = Data @ P                  # t × q
cov_small = Y.mT @ Y         # q × q
# eigen decompose the small q×q covariance
eigvals, eigvecs = pt.linalg.eigh(cov_small)
eigvals = eigvals.flip(-1)
eigvecs = eigvecs.flip(-1)
# lift back to d-dimensional space
U_full = P @ eigvecs          # d × q
ev("eigh2")


# S = eigvals.sqrt()

# relative reconstruction error on the covariance: ||C - U diag(eigvals) U^T|| / ||C||
approx = U_full * eigvals.unsqueeze(-2) @ U_full.transpose(-1, -2)
rel_error = pt.linalg.norm(G - approx, dim=(-2, -1)) / pt.linalg.norm(G, dim=(-2, -1))
print(f"Relative reconstruction error: {rel_error.mean():.6f}")


# %%
Data.shape
# %%
m = pt.randn(10, 400, 400, device="cuda")
start_time = time.time()
# pt.linalg.svd(m, full_matrices=False)
# pt.linalg.eigh(m)
pt.linalg.inv(m)
ev(0)




# noise:
# U_full += 0.03 * pt.randn(U_full.shape, device="cuda")


# x_proj = x @ P                          # project to q-dim
# mahal_proj = x_proj @ cov_small_inv     # Mahalanobis in small space
# mahal_dir = mahal_proj @ P.T            # lift back to d-dim
# # wait, but we lose info here! previously we preserved the info outside of our 400 dims

# torch.linalg.inv(cov_small)



# %%

# # ! estimate eigvals.min() somehow?
# or:
# Power iteration for max eigenvalue of the inverse (= 1/λ_min): just a few iterations of v = cov_small_inv @ v; v /= v.norm(), then λ_min ≈ 1 / (v @ cov_small_inv @ v). A few iterations of q×q matmuls — microseconds.



# precompute once (after getting P and cov_small):
eigvals_min = eigvals.min()  # you do need this one scalar — but eigh of q×q is cheap
# OR approximate it from trace / q, or just use a small constant

# actually, you need the reweighting matrix in the small space:
# M_small = I - cov_small_inv * eigvals_min
# which requires cov_small_inv

cov_small_inv = torch.linalg.inv(cov_small)  # or cholesky_inverse

# online: for each x
x_proj = centered @ P                                    # project to q-dim
correction_proj = x_proj - eigvals_min * (x_proj @ cov_small_inv)  # reweight in small space
correction = correction_proj @ P.T                        # lift back
result = centered - correction                            # only touches the P subspace