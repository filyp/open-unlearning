import torch as pt
# from torch_incremental_pca import IncrementalPCA
# from trainer.unlearn.repselect.online_covariance import OnlineCovariance, OnlineCovarianceSimple
# todo uninstall welford_torch


def _proj_to_mahal_dirs(centered, mahal_dirs):
    mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
    proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    return proj_strenghts * mahal_dirs_norm


# todo: watch out for never-selected experts - detect it through None cov?
# wait, we already have that!
# but maybe have some assert that we gathered enough data? in add_vecs.
#     not assert, just ignore the reprocessing then?

class CovCollapser:
    """
    In the pass (before first call to process_saved_vecs), we prepare an accurate P (that captures the most important directions).
    In the second pass, using that P, we calculate a covariance matrix in that most important subspace.
    In the third pass, we can use the inverted covariance matrix together with P used in the previous pass, to collapse vecs onto the Mahalanobis directions.
    """
    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use
        self.cov = None
        self.total_vecs = 0
        self.P = None
        self.P_is_ready = False
        self.inv_cov_valid = False
        self.mean = None
        self._mean_count = 0
        self._first_pass = True
        self._full_cov = None

    def add_vecs(self, vecs):
        vecs = vecs.float()
        n = vecs.shape[0]
        if self.mean is None:
            dim = vecs.shape[1]
            self.mean = pt.zeros(dim, dtype=pt.float32, device=vecs.device)

        # Welford online mean update
        self._mean_count += n
        self.mean += (vecs - self.mean).sum(0) / self._mean_count
        centered = vecs - self.mean

        if self._first_pass:
            # Accumulate full (D, D) covariance for accurate initial P via svd_lowrank
            if self._full_cov is None:
                dim = centered.shape[1]
                self._full_cov = pt.zeros(dim, dim, dtype=pt.bfloat16, device=vecs.device)
            self._full_cov += pt.einsum("ni,nj->ij", centered.bfloat16(), centered.bfloat16())
            return

        self.future_P += centered.mT @ (centered @ self.P.float())

        if self.P_is_ready:
            Y = centered @ self.P.float()
            if self.cov is None:
                dim = Y.shape[1]
                self.cov = pt.zeros(dim, dim, dtype=pt.float32, device=vecs.device)
            self.cov += Y.mT @ Y
            self.total_vecs += n

    def process_saved_vecs(self):
        if self._first_pass:
            # Use full covariance to get accurate initial P
            self._first_pass = False
            if self._full_cov is not None:
                _, _, V = pt.svd_lowrank(self._full_cov.float(), q=self.PCs_to_use)
                self.P = V.bfloat16()  # (D, k)
                self.future_P = pt.zeros_like(V)
                self._full_cov = None
            # self.mean = pt.zeros_like(self.mean)
            self._mean_count = self._mean_count // 2
            self.P_is_ready = True
            return

        # Extract distribution stats from online covariance
        if self.cov is not None:
            if self.total_vecs > 500:  # some experts may have too few vectors to invert covariance matrix
                self.inv_cov_valid = True

                eps = self.cov.diag().mean() * 1e-3
                self.inv_cov = pt.linalg.inv(self.cov + eps * pt.eye(self.PCs_to_use, device=self.cov.device))

                # estimate eigvals_min using power iteration
                v = pt.randn(self.PCs_to_use, device=self.inv_cov.device)
                for _ in range(10):
                    v = self.inv_cov @ v
                    v = v / v.norm()
                eigvals_min = 1.0 / (v @ self.inv_cov @ v)
                self.inv_cov *= eigvals_min

            else:
                self.inv_cov_valid = False
                self.inv_cov = pt.zeros_like(self.cov)

            self.old_P = self.P
            self.old_mean = self.mean.clone()
            self.cov = None
            self.total_vecs = 0

        self.mean = pt.zeros_like(self.mean)
        self._mean_count = 0
        self.P = pt.linalg.qr(self.future_P).Q.bfloat16()
        self.P_is_ready = True

    def collapse(self, vecs):
        orig_dtype = vecs.dtype
        centered = vecs.float() - self.old_mean
        old_P = self.old_P.float()

        x_proj = centered @ old_P
        correction_proj = x_proj - (x_proj @ self.inv_cov)
        correction = correction_proj @ old_P.T  # lift back
        mahal_dirs = centered - correction  # only touches the P subspace

        return _proj_to_mahal_dirs(centered, mahal_dirs).to(orig_dtype)


class BatchedCovCollapser:
    """Wraps E individual CovStoringCollapsers, exposes batched collapse via _grouped_mm."""

    eig_val: pt.Tensor  # (E, k)
    eig_vec: pt.Tensor  # (E, D, k)

    def __init__(self, PCs_to_use: int, num_experts: int):
        self.collapsers = [CovCollapser(PCs_to_use) for _ in range(num_experts)]
        self.num_experts = num_experts

    def add_vecs(self, expert_idx: int, vecs: pt.Tensor):
        self.collapsers[expert_idx].add_vecs(vecs)

    def process_saved_vecs(self):
        # note: tried batching this too, but it did not help and was very complex
        for c in self.collapsers:
            c.process_saved_vecs()
        
        if not hasattr(self.collapsers[0], "inv_cov"):
            return  # inv_cov not ready yet

        # Stack per-expert stats into batched tensors; keep previous for experts with no new data
        inv_covs, old_Ps, old_means, inv_cov_valid = [], [], [], []
        for i, c in enumerate(self.collapsers):
            inv_covs.append(c.inv_cov)
            old_Ps.append(c.old_P)
            old_means.append(c.old_mean)
            inv_cov_valid.append(pt.tensor(c.inv_cov_valid, device=c.inv_cov.device))
        self.inv_covs = pt.stack(inv_covs) # (E, k, k)
        self.old_Ps = pt.stack(old_Ps) # (E, d, k)
        self.old_means = pt.stack(old_means) # (E, D)
        self.inv_cov_valid = pt.stack(inv_cov_valid) # (E,)

    def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
        orig_dtype = vecs_sorted.dtype
        vecs_sorted = vecs_sorted.float()
        old_Ps = self.old_Ps.float()

        # center per-expert
        expert_ids = pt.bucketize(pt.arange(vecs_sorted.shape[0], device=vecs_sorted.device), offsets)
        centered = vecs_sorted - self.old_means[expert_ids]

        # get mahalanobis directions
        # (S, d) @ (E, d, k) -> (S, k)
        x_proj = pt._grouped_mm(centered, old_Ps, offs=offsets)
        # (S, k) @ (E, k, k) -> (S, k)
        correction_proj = x_proj - pt._grouped_mm(x_proj, self.inv_covs, offs=offsets)
        # (S, k) @ (E, k, d) -> (S, d)
        correction = pt._grouped_mm(correction_proj, old_Ps.mT, offs=offsets)
        mahal_dirs = centered - correction  # only touches the P subspace
        
        collapsed_vecs = _proj_to_mahal_dirs(centered, mahal_dirs)

        # zero out tokens from experts with invalid inv_cov
        # collapsed_vecs[~self.inv_cov_valid[expert_ids]] = 0  # todo reenable
        assert not collapsed_vecs.isnan().any()

        return collapsed_vecs.to(orig_dtype)


# # just as bad!
# class BatchedCovCollapser:
#     """Uses eigh on small cov (via P projection) to get full-space eigenvectors."""

#     def __init__(self, PCs_to_use: int, num_experts: int):
#         self.collapsers = [CovCollapser(PCs_to_use) for _ in range(num_experts)]
#         self.num_experts = num_experts

#     def add_vecs(self, expert_idx: int, vecs: pt.Tensor):
#         self.collapsers[expert_idx].add_vecs(vecs)

#     def process_saved_vecs(self):
#         eig_vals_list, eig_vecs_list, valid_list = [], [], []
#         k = self.collapsers[0].PCs_to_use
#         device = None

#         for c in self.collapsers:
#             # Update P (power iteration + QR) without discarding cov
#             if c.cov is not None and c.total_vecs > 100:
#                 old_P = c.P.float()  # (D, k) — the P used to build this cov
#                 cov_small = c.cov    # (k, k) already float32

#                 # eigh on small cov: cov_small = Q diag(λ) Q^T
#                 eigvals, eigvecs_small = pt.linalg.eigh(cov_small)
#                 # eigh returns ascending order; flip to descending
#                 eigvals = eigvals.flip(-1)
#                 eigvecs_small = eigvecs_small.flip(-1)

#                 # lift to full D-space: U_full = P @ eigvecs_small, shape (D, k)
#                 eig_vecs_full = old_P @ eigvecs_small

#                 eig_vals_list.append(eigvals)
#                 eig_vecs_list.append(eig_vecs_full)
#                 valid_list.append(True)
#                 if device is None:
#                     device = cov_small.device
#             else:
#                 # placeholder — will be zeroed out
#                 if device is None:
#                     device = c.P.device if c.P is not None else "cuda"
#                 eig_vals_list.append(pt.ones(k, device=device))
#                 eig_vecs_list.append(pt.zeros(c.PCs_to_use, k, device=device) if c.P is None
#                                      else pt.zeros_like(c.P.float()))
#                 valid_list.append(False)

#             # Now do the normal P update (same as CovCollapser.process_saved_vecs)
#             c.cov = None
#             c.total_vecs = 0
#             c.P = pt.linalg.qr(c.future_P).Q.bfloat16()
#             c.P_is_ready = True

#         self.eig_val = pt.stack(eig_vals_list)    # (E, k)
#         self.eig_vec = pt.stack(eig_vecs_list)    # (E, D, k)
#         self.valid = pt.tensor(valid_list, device=device)

#     def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
#         orig_dtype = vecs_sorted.dtype
#         vecs_sorted = vecs_sorted.float()
#         S = vecs_sorted.shape[0]
#         device = vecs_sorted.device

#         expert_ids = pt.bucketize(pt.arange(S, device=device), offsets)

#         # Project onto eigenvectors: (S, D) × (E, D, k) → (S, k)
#         projected = pt._grouped_mm(vecs_sorted, self.eig_vec, offs=offsets)

#         # Eigenvalue reweighting: scale = 1 - λ_min/λ_i
#         eig_val_tok = self.eig_val[expert_ids]  # (S, k)
#         eig_val_min = eig_val_tok.min(dim=1, keepdim=True).values
#         proj_diff = projected * (1 - eig_val_min / eig_val_tok)

#         # Back-project: (S, k) × (E, k, D) → (S, D)
#         eig_vec_T = self.eig_vec.transpose(-2, -1).contiguous()
#         correction = pt._grouped_mm(proj_diff, eig_vec_T, offs=offsets)
#         mahal_dirs = vecs_sorted - correction

#         collapsed_vecs = _proj_to_mahal_dirs(vecs_sorted, mahal_dirs)

#         # zero out invalid experts
#         collapsed_vecs[~self.valid[expert_ids]] = 0

#         return collapsed_vecs.to(orig_dtype)


# def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
#     """Batched collapse using _grouped_mm. vecs_sorted: (S, D), offsets: (E,)."""
#     dtype = vecs_sorted.dtype
#     S = vecs_sorted.shape[0]
#     device = vecs_sorted.device

#     # Expand per-expert mean to per-token for centering
#     expert_ids = pt.bucketize(pt.arange(S, device=device), offsets)
#     centered = vecs_sorted - self.mean[expert_ids]  # (S, D)

#     # Project onto eigenvectors: (S, D) × (E, D, k) → (S, k)
#     projected = pt._grouped_mm(centered, self.eig_vec, offs=offsets)

#     # Eigenvalue reweighting (per-token, using expanded eig_val)
#     eig_val_tok = self.eig_val[expert_ids]  # (S, k) — small
#     eig_val_tok = eig_val_tok / eig_val_tok.min(dim=1, keepdim=True).values
#     proj_diff = projected - projected / eig_val_tok

#     # Back-project: (S, k) × (E, k, D) → (S, D)
#     eig_vec_T = self.eig_vec.transpose(-2, -1).contiguous()
#     correction = pt._grouped_mm(proj_diff, eig_vec_T, offs=offsets)
#     mahal_dirs = centered - correction

#     result = _proj_to_mahal_dirs(centered, mahal_dirs)
#     assert result.shape == vecs_sorted.shape
#     return result.to(dtype)











# def _get_mahal_dirs(centered, eig_val, eig_vec):
#     # Compute Mahalanobis directions using eigendecomposition
#     projected = centered @ eig_vec  # (N, D)
#     proj_diff = projected - projected / (eig_val / eig_val.min())
#     # neg = projected / (eig_val / eig_val.min())
#     # neg[:, -24:] = 0
#     # proj_diff = projected - neg
#     return centered - proj_diff @ eig_vec.T
# class IncrementalPCACollapser:
#     def __init__(self, PCs_to_use: int):
#         self.PCs_to_use = PCs_to_use
#         self.accumulator = None
#         self.ipca = IncrementalPCA(n_components=PCs_to_use, gram=True)

#     def add_vecs(self, vecs):
#         """In addition to partial_fit, it fill also accumulate the vecs if needed"""
#         if self.accumulator is not None:
#             vecs = pt.cat([self.accumulator, vecs])
#             self.accumulator = None

#         if vecs.shape[0] < self.ipca.n_components:  # too few vecs, so accumulate
#             self.accumulator = vecs
#         else:  # enough vecs
#             vecs = vecs.clone()  # ipca.partial_fit may modify the input in-place
#             self.ipca.partial_fit(vecs)

#     def process_saved_vecs(self):
#         pass  # components are updated online in partial_fit

#     def collapse(self, vecs):
#         eig_vec = self.ipca.components_.T  # (n_features, n_components)
#         eig_val = self.ipca.explained_variance_  # (n_components,)
#         centered = vecs - self.ipca.mean_
#         assert centered.dtype == pt.float32
#         mahal_dirs = _get_mahal_dirs(centered, eig_val, eig_vec)
#         return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)



class CovCollapserSimple:
    """Simplified but much slower version of CovCollapser, using svd."""
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use
        self.cov = None

    def add_vecs(self, vecs):
        if self.cov is None:
            D = vecs.shape[1]
            self.cov = pt.zeros(D, D, dtype=pt.bfloat16, device=vecs.device)
        vecs = vecs.to(pt.bfloat16)
        self.cov += pt.einsum("ni,nj->ij", vecs, vecs)

    def process_saved_vecs(self):
        # Extract distribution stats from online covariance
        self.inv_cov = pt.linalg.inv(self.cov.float()).bfloat16()

        cov = self.cov.to(pt.float32)

        _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use)

        self.eig_val = S  # top-k eigenvalues (largest first)
        self.eig_vec = V  # (D, k)
        self.cov = None

    def collapse(self, vecs):
        centered = vecs.to(pt.float32)

        # Compute Mahalanobis directions using eigendecomposition
        projected = centered @ self.eig_vec  # (N, D)
        proj_diff = projected - projected / (self.eig_val / self.eig_val.min())
        mahal_dirs = centered - proj_diff @ self.eig_vec.T

        return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)


# # this variant relies on CovCollapserSimple, which is much slower
# class BatchedCovCollapser:
#     """Wraps E individual CovStoringCollapsers, exposes batched collapse via _grouped_mm."""

#     mean: pt.Tensor     # (E, D)
#     eig_val: pt.Tensor  # (E, k)
#     eig_vec: pt.Tensor  # (E, D, k)

#     def __init__(self, PCs_to_use: int, num_experts: int):
#         self.collapsers = [CovCollapser(PCs_to_use) for _ in range(num_experts)]
#         self.num_experts = num_experts

#     def add_vecs(self, expert_idx: int, vecs: pt.Tensor):
#         self.collapsers[expert_idx].add_vecs(vecs)

#     def process_saved_vecs(self):
#         # note: tried batching this too, but it did not help and was very complex
#         for c in self.collapsers:
#             c.process_saved_vecs()
#         # Stack per-expert stats into batched tensors; keep previous for experts with no new data
#         means, eig_vals, eig_vecs = [], [], []
#         for i, c in enumerate(self.collapsers):
#             assert hasattr(c, "mean"), f"Expert {i} has no stats — was it never routed any tokens?"
#             means.append(c.mean)
#             eig_vals.append(c.eig_val)
#             eig_vecs.append(c.eig_vec)
#         self.mean = pt.stack(means)       # (E, D)
#         self.eig_val = pt.stack(eig_vals) # (E, k)
#         self.eig_vec = pt.stack(eig_vecs) # (E, D, k)

#     def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
#         """Batched collapse using _grouped_mm. vecs_sorted: (S, D), offsets: (E,)."""
#         dtype = vecs_sorted.dtype
#         S = vecs_sorted.shape[0]
#         device = vecs_sorted.device

#         # Expand per-expert mean to per-token for centering
#         expert_ids = pt.bucketize(pt.arange(S, device=device), offsets)
#         centered = vecs_sorted - self.mean[expert_ids]  # (S, D)

#         # Project onto eigenvectors: (S, D) × (E, D, k) → (S, k)
#         projected = pt._grouped_mm(centered, self.eig_vec, offs=offsets)

#         # Eigenvalue reweighting (per-token, using expanded eig_val)
#         eig_val_tok = self.eig_val[expert_ids]  # (S, k) — small
#         eig_val_tok = eig_val_tok / eig_val_tok.min(dim=1, keepdim=True).values
#         proj_diff = projected - projected / eig_val_tok

#         # Back-project: (S, k) × (E, k, D) → (S, D)
#         eig_vec_T = self.eig_vec.transpose(-2, -1).contiguous()
#         correction = pt._grouped_mm(proj_diff, eig_vec_T, offs=offsets)
#         mahal_dirs = centered - correction

#         result = _proj_to_mahal_dirs(centered, mahal_dirs)
#         assert result.shape == vecs_sorted.shape
#         return result.to(dtype)

