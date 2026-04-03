import torch as pt

from trainer.unlearn.repselect.online_covariance import (
    BatchedOnlineCovariance,
    OnlineCovariance,
)


def _proj_to_mahal_dirs(centered, mahal_dirs):
    mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
    proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    return proj_strenghts * mahal_dirs_norm


class SVDCollapser:
    mean: pt.Tensor
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use
        self.online_cov = OnlineCovariance(dtype=pt.bfloat16)

    def add_vecs(self, vecs):
        self.online_cov.add_vecs(vecs)

    def fit(self):
        # Extract distribution stats from online covariance
        cov = self.online_cov.get_cov().float()
        self.mean = self.online_cov.mean.float().clone()

        # _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use, niter=0)
        # adapted from svd_lowrank, but niter=0, and provides hot-start (eig_vec)
        init = (
            self.eig_vec  # provides hot-start, instead of random initialization
            if hasattr(self, "eig_vec")
            else pt.randn(
                cov.shape[0], self.PCs_to_use, dtype=pt.float32, device=cov.device
            )
        )
        Q = pt.linalg.qr(cov @ init).Q
        # for _ in range(0):  # only needed if eig_vals are flat, which is not the case for LLM activations
        #     Q = pt.linalg.qr(cov @ Q).Q
        B = Q.mT @ cov
        _, S, Vh = pt.linalg.svd(B, full_matrices=False)
        V = Vh.mT

        self.eig_val = S / S.min()  # normalized so min=1
        self.eig_vec = V  # (D, k)

        # Reset online covariance for next epoch
        self.online_cov = OnlineCovariance(dtype=pt.bfloat16)

    def collapse(self, vecs):
        original_dtype = vecs.dtype
        vecs = vecs.float()
        centered = vecs - self.mean

        # get Mahalanobis directions
        projected = centered @ self.eig_vec  # (N, D)
        proj_diff = projected - projected / self.eig_val  # assumes eig_val.min() == 1
        mahal_dirs = centered - proj_diff @ self.eig_vec.T

        return _proj_to_mahal_dirs(centered, mahal_dirs).to(original_dtype)


class InvSmallCovCollapser:
    """
    In the first pass (before first call to fit), we prepare an accurate P that captures the most important directions.
    In the second pass, using that P, we calculate a small covariance matrix in that subspace.
    In the third pass, we use the inverted covariance together with the P from the previous pass to collapse vecs onto the Mahalanobis directions.
    """

    P: pt.Tensor         # (D, k) current projection matrix (refined each epoch)
    old_P: pt.Tensor     # (D, k) projection used during the last cov-accumulation pass
    future_P: pt.Tensor  # (D, k) accumulated for next P refinement
    mean_proj: pt.Tensor  # (k,) mean of projected vectors from last cov-accumulation pass
    inv_cov: pt.Tensor   # (k, k) = eigvals_min * inv(small_cov)

    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use
        self.P_is_ready = False

    def add_vecs(self, vecs):
        vecs = vecs.float()
        if not hasattr(self, "P"):
            D = vecs.shape[1]
            _rand = pt.randn(D, self.PCs_to_use, dtype=pt.float32, device=vecs.device)
            self.P = pt.linalg.qr(_rand).Q
            self.future_P = pt.zeros_like(self.P)

        self.future_P += vecs.mT @ (vecs @ self.P)

        if self.P_is_ready:
            Y = vecs @ self.P  # (N, k)
            self.small_online_cov.add_vecs(Y)

    def fit(self):
        if hasattr(self, "small_online_cov"):
            small_cov = self.small_online_cov.get_cov().float()
            inv_cov = pt.linalg.inv(small_cov)
            self.old_P = self.P
            self.mean_proj = self.small_online_cov.mean.float().clone()

            # estimate eigvals_min using power iteration on inv_cov
            v = pt.randn(self.PCs_to_use, dtype=pt.float32, device=inv_cov.device)
            for _ in range(10):
                v = inv_cov @ v
                v = v / v.norm()
            eigvals_min = 1.0 / (v @ inv_cov @ v)
            self.inv_cov = inv_cov * eigvals_min  # absorb scaling so collapse needs no separate eigvals_min

        self.P = pt.linalg.qr(self.future_P).Q
        self.future_P = pt.zeros_like(self.P)
        self.small_online_cov = OnlineCovariance(dtype=pt.float32)
        self.P_is_ready = True

    def collapse(self, vecs):
        original_dtype = vecs.dtype
        vecs = vecs.float()

        mean = self.mean_proj @ self.old_P.mT  # (D,) — P-subspace component of the mean
        vecs_centered = vecs - mean  # (N, D)

        centered_proj = vecs_centered @ self.old_P  # (N, k)
        correction_proj = centered_proj - (centered_proj @ self.inv_cov)  # (N, k)
        correction = correction_proj @ self.old_P.mT  # (N, D)
        mahal_dirs = vecs_centered - correction

        return _proj_to_mahal_dirs(vecs_centered, mahal_dirs).to(original_dtype)


class BatchedCovCollapser:
    """Single BatchedOnlineCovariance for all experts; exposes batched collapse via _grouped_mm."""

    mean: pt.Tensor  # (E, D)
    eig_val: pt.Tensor  # (E, k)
    eig_vec: pt.Tensor  # (E, D, k)

    def __init__(self, PCs_to_use: int, num_experts: int):
        self.PCs_to_use = PCs_to_use
        self.num_experts = num_experts
        self.online_cov = BatchedOnlineCovariance(self.num_experts)

    def add_vecs(self, vecs: pt.Tensor, offsets: pt.Tensor):
        self.online_cov.add_vecs(vecs, offsets)

    def fit(self):
        device = self.online_cov.device
        cov_full = self.online_cov.get_cov().float()
        self.mean = self.online_cov.mean.float().clone()
        D = cov_full.shape[-1]
        k = self.PCs_to_use
        init = (
            self.eig_vec
            if hasattr(self, "eig_vec")
            else pt.randn(self.num_experts, D, k, device=device, dtype=pt.float32)
        )

        Q = pt.linalg.qr(cov_full @ init).Q
        B = Q.mT @ cov_full
        _, S, Vh = pt.linalg.svd(B, full_matrices=False)
        V = Vh.mT

        self.eig_vec = V
        self.eig_val = S / S.min(dim=1, keepdim=True).values

        # reset online covariance for next epoch
        self.online_cov = BatchedOnlineCovariance(self.num_experts)

    def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
        """Batched collapse using _grouped_mm. vecs_sorted: (S, D), offsets: (E,)."""
        original_dtype = vecs_sorted.dtype
        vecs_sorted = vecs_sorted.float()
        S = vecs_sorted.shape[0]
        device = vecs_sorted.device

        # Expand per-expert mean to per-token for centering
        expert_ids = pt.bucketize(pt.arange(S, device=device), offsets, right=True)
        centered = vecs_sorted - self.mean[expert_ids]  # (S, D)

        # Project onto eigenvectors: (S, D) × (E, D, k) → (S, k)
        projected = pt._grouped_mm(centered, self.eig_vec, offs=offsets)

        # Eigenvalue reweighting (per-token, using expanded eig_val)
        eig_val_tok = self.eig_val[expert_ids]  # (S, k) — already normalized (min=1)
        proj_diff = projected - projected / eig_val_tok

        # Back-project: (S, k) × (E, k, D) → (S, D)
        eig_vec_T = self.eig_vec.mT.contiguous()
        correction = pt._grouped_mm(proj_diff, eig_vec_T, offs=offsets)
        mahal_dirs = centered - correction

        result = _proj_to_mahal_dirs(centered, mahal_dirs)
        assert result.shape == vecs_sorted.shape
        return result.to(original_dtype)


# # print some eig_val statistics
# max_eig_val = S.max(dim=1).values
# min_eig_val = S.min(dim=1).values
# ratio = max_eig_val / min_eig_val
# print(pt.stack([max_eig_val, min_eig_val, ratio], dim=1))



# class InvCovCollapser:
#     "Surprisingly, it works just as fast as SVDCollapser, so SVD was no longer the bottleneck."
#     mean: pt.Tensor
#     P: pt.Tensor  # (D, k) orthonormal projection matrix
#     inv_small_cov: pt.Tensor  # (k, k) = eigvals_min * inv(P.T @ cov @ P)

#     def __init__(self, PCs_to_use: int):
#         self.PCs_to_use = PCs_to_use
#         self.online_cov = OnlineCovariance(dtype=pt.bfloat16)

#     def add_vecs(self, vecs):
#         self.online_cov.add_vecs(vecs)

#     def fit(self):
#         self.mean = self.online_cov.mean.float()
#         cov = self.online_cov.get_cov().float()

#         # Initialize or refine projection matrix via one subspace-iteration step (no SVD)
#         init = (
#             self.P
#             if hasattr(self, "P")
#             else pt.randn(
#                 cov.shape[0], self.PCs_to_use, dtype=pt.float32, device=cov.device
#             )
#         )
#         self.P = pt.linalg.qr(cov @ init).Q  # (D, k)

#         # Compute and invert the k×k projected covariance
#         small_cov = self.P.mT @ cov @ self.P  # (k, k)
#         self.inv_small_cov = pt.linalg.inv(small_cov)  # (k, k)

#         # Estimate min eigenvalue of small_cov via power iteration on its inverse
#         v = pt.randn(self.PCs_to_use, dtype=pt.float32, device=cov.device)
#         for _ in range(10):
#             v = self.inv_small_cov @ v
#             v = v / v.norm()
#         eigvals_min = (1.0 / (v @ self.inv_small_cov @ v)).item()
#         self.inv_small_cov *= eigvals_min

#         self.online_cov = OnlineCovariance(dtype=pt.bfloat16)

#     def collapse(self, vecs):
#         original_dtype = vecs.dtype
#         vecs = vecs.float()
#         centered = vecs - self.mean

#         x_proj = centered @ self.P  # (N, k)
#         correction_proj = x_proj - (x_proj @ self.inv_small_cov)  # (N, k)
#         correction = correction_proj @ self.P.mT  # (N, D)
#         mahal_dirs = centered - correction

#         return _proj_to_mahal_dirs(centered, mahal_dirs).to(original_dtype)




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

#     def fit(self):
#         pass  # components are updated online in partial_fit

#     def collapse(self, vecs):
#         eig_vec = self.ipca.components_.T  # (n_features, n_components)
#         eig_val = self.ipca.explained_variance_  # (n_components,)
#         centered = vecs - self.ipca.mean_
#         assert centered.dtype == pt.float32
#         mahal_dirs = _get_mahal_dirs(centered, eig_val, eig_vec)
#         return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)


#         _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use, niter=0)
#         self.eig_val = S  # top-k eigenvalues (largest first)
#         self.eig_vec = V  # (D, k)
#         self._reset_vecs()


# class PartiallyBatchedCovCollapser:
#     """Wraps E individual CovStoringCollapsers, exposes batched collapse via _grouped_mm.

#     A partially batched version of BatchedCovCollapser, that may be slower, but is simpler.
#     """

#     mean: pt.Tensor  # (E, D)
#     eig_val: pt.Tensor  # (E, k)
#     eig_vec: pt.Tensor  # (E, D, k)

#     def __init__(self, PCs_to_use: int, num_experts: int):
#         self.collapsers = [CovCollapser(PCs_to_use) for _ in range(num_experts)]
#         self.num_experts = num_experts

#     def add_vecs(self, vecs: pt.Tensor, offsets: pt.Tensor, num_experts: int):
#         ends = offsets.tolist()
#         starts = [0] + ends[:-1]
#         # Accumulate collapser stats (per-expert, variable token counts)
#         for expert_idx in range(num_experts):
#             start, end = starts[expert_idx], ends[expert_idx]
#             if start == end:
#                 continue
#             self.collapsers[expert_idx].add_vecs(vecs[start:end])

#     def fit(self):
#         # note: tried batching this too, but it did not help and was very complex
#         for c in self.collapsers:
#             c.fit()
#         # Stack per-expert stats into batched tensors; keep previous for experts with no new data
#         # for i, c in enumerate(self.collapsers):
#         #     assert hasattr(c, "mean"), f"Expert {i} has no stats — was it never routed any tokens?"
#         self.mean = pt.stack([c.mean for c in self.collapsers])  # (E, D)
#         self.eig_val = pt.stack([c.eig_val for c in self.collapsers])  # (E, k)
#         self.eig_vec = pt.stack([c.eig_vec for c in self.collapsers])  # (E, D, k)

#     def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
#         """Batched collapse using _grouped_mm. vecs_sorted: (S, D), offsets: (E,)."""
#         dtype = vecs_sorted.dtype
#         S = vecs_sorted.shape[0]
#         device = vecs_sorted.device

#         # Expand per-expert mean to per-token for centering
#         expert_ids = pt.bucketize(pt.arange(S, device=device), offsets, right=True)
#         centered = vecs_sorted - self.mean[expert_ids]  # (S, D)

#         # Project onto eigenvectors: (S, D) × (E, D, k) → (S, k)
#         projected = pt._grouped_mm(centered, self.eig_vec, offs=offsets)

#         # Eigenvalue reweighting (per-token, using expanded eig_val)
#         eig_val_tok = self.eig_val[expert_ids]  # (S, k) — already normalized (min=1)
#         proj_diff = projected - projected / eig_val_tok

#         # Back-project: (S, k) × (E, k, D) → (S, D)
#         eig_vec_T = self.eig_vec.mT.contiguous()
#         correction = pt._grouped_mm(proj_diff, eig_vec_T, offs=offsets)
#         mahal_dirs = centered - correction

#         result = _proj_to_mahal_dirs(centered, mahal_dirs)
#         assert result.shape == vecs_sorted.shape
#         return result.to(dtype)
