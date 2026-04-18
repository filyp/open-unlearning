import torch as pt

from trainer.unlearn.repselect.online_covariance import (
    BatchedOnlineCovariance,
    OnlineCovariance,
)


class InvSmallCovCollapser:
    """
    In the first pass (before first call to fit), we prepare an accurate P that captures the most important directions.
    In the second pass, we compute the small second-moment matrix from cov_P directly (P.T @ cov_P = P.T @ X.T @ X @ P),
    and use its inverse together with P to collapse vecs onto the Mahalanobis directions.
    """

    P: pt.Tensor  # (D, k) current projection matrix (refined each epoch)
    cov_P: pt.Tensor  # (D, k) accumulated X.T @ X @ P
    inv_cov: pt.Tensor  # (k, k) = eigvals_min * inv(P.T @ X.T @ X @ P)

    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use

    def add_vecs(self, vecs):
        vecs = vecs.float()
        if not hasattr(self, "P"):
            D = vecs.shape[1]
            _rand = pt.randn(D, self.PCs_to_use, dtype=pt.float32, device=vecs.device)
            self.P = pt.linalg.qr(_rand).Q
            self.cov_P = pt.zeros_like(self.P)

        self.cov_P += vecs.mT @ (vecs @ self.P)

    def fit(self):
        small_cov = self.P.mT @ self.cov_P  # (k, k) = P.T @ X.T @ X @ P, exact

        # refine P
        old_P = self.P
        self.P = pt.linalg.qr(self.cov_P).Q
        self.cov_P = pt.zeros_like(self.P)

        # adjust small_cov, to be expressed in the new basis
        _reprojection = old_P.mT @ self.P
        small_cov = _reprojection.mT @ small_cov @ _reprojection

        # regularize and invert
        _scale = small_cov.diagonal().amax()
        small_cov.diagonal().add_(1e-6 * _scale.clamp(min=1.0))
        inv_cov = pt.linalg.inv(small_cov)

        # estimate eigvals_min using power iteration on inv_cov
        v = pt.randn(self.PCs_to_use, dtype=pt.float32, device=self.P.device)
        for _ in range(10):
            v = inv_cov @ v
            v = v / v.norm()
        eigvals_min = 1.0 / (v @ inv_cov @ v)
        # absorb scaling so collapse needs no separate eigvals_min
        self.inv_cov = inv_cov * eigvals_min

    def collapse(self, vecs):
        original_dtype = vecs.dtype
        vecs = vecs.float()

        x_proj = vecs @ self.P  # (N, k)
        correction_proj = x_proj - (x_proj @ self.inv_cov)  # (N, k)
        correction = correction_proj @ self.P.mT  # (N, D)
        mahal_dirs = vecs - correction

        return mahal_dirs.to(original_dtype)


class BatchedInvSmallCovCollapser:
    """Batched version of InvSmallCovCollapser for MoE experts."""

    P: pt.Tensor  # (E, D, k)
    cov_P: pt.Tensor  # (E, D, k) accumulated X.T @ X @ P
    inv_cov: pt.Tensor  # (E, k, k) = eigvals_min * inv(small_cov), per expert

    def __init__(self, PCs_to_use: int, num_experts: int):
        self.PCs_to_use = PCs_to_use
        self.num_experts = num_experts

    def add_vecs(self, vecs: pt.Tensor, offsets: pt.Tensor):
        vecs = vecs.float()
        if not hasattr(self, "P"):
            E, D, k = self.num_experts, vecs.shape[1], self.PCs_to_use
            _rand = pt.randn(E, D, k, dtype=pt.float32, device=vecs.device)
            self.P = pt.linalg.qr(_rand).Q  # (E, D, k)
            self.cov_P = pt.zeros_like(self.P)

        Y = pt._grouped_mm(vecs, self.P, offs=offsets)  # (S, k)
        self.cov_P += pt._grouped_mm(vecs.mT, Y, offs=offsets)  # (E, D, k)

    def fit(self):
        k = self.PCs_to_use
        small_cov = self.P.mT @ self.cov_P  # (E, k, D) @ (E, D, k) = (E, k, k)

        # refine P
        old_P = self.P
        self.P = pt.linalg.qr(self.cov_P).Q  # (E, D, k)
        self.cov_P = pt.zeros_like(self.P)

        # adjust small_cov to be expressed in the new basis
        _reprojection = old_P.mT @ self.P  # (E, k, k)
        small_cov = _reprojection.mT @ small_cov @ _reprojection

        # regularize and invert
        _scale = small_cov.diagonal(dim1=-2, dim2=-1).amax(dim=-1, keepdim=True)
        small_cov.diagonal(dim1=-2, dim2=-1).add_(1e-6 * _scale.clamp(min=1.0))
        inv_cov = pt.linalg.inv(small_cov)  # (E, k, k)

        # estimate eigvals_min per expert via batched power iteration on inv_cov
        v = pt.randn(self.num_experts, k, dtype=pt.float32, device=self.P.device)
        for _ in range(10):
            v = (inv_cov @ v.unsqueeze(-1)).squeeze(-1)  # (E, k)
            v = v / v.norm(dim=1, keepdim=True)
        v = v.unsqueeze(-1)  # (E, k, 1)
        eigvals_min = 1.0 / (v.mT @ inv_cov @ v).squeeze(-1).squeeze(-1)  # (E,)
        self.inv_cov = inv_cov * eigvals_min[:, None, None]

    def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
        """Batched collapse. vecs_sorted: (S, D), offsets: (E,)."""
        original_dtype = vecs_sorted.dtype
        vecs_sorted = vecs_sorted.float()

        x_proj = pt._grouped_mm(vecs_sorted, self.P, offs=offsets)  # (S, k)
        correction_proj = x_proj - pt._grouped_mm(x_proj, self.inv_cov, offs=offsets)
        _up_proj = self.P.mT
        correction = pt._grouped_mm(correction_proj, _up_proj, offs=offsets)  # (S, D)
        mahal_dirs = vecs_sorted - correction

        assert mahal_dirs.shape == vecs_sorted.shape
        return mahal_dirs.to(original_dtype)


########################################################################################


# def _proj_to_mahal_dirs(centered, mahal_dirs):
#     norms = mahal_dirs.norm(dim=1, keepdim=True)
#     valid = norms > 0
#     mahal_dirs_norm = mahal_dirs / norms.clamp(min=1e-30)
#     proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
#     result = proj_strenghts * mahal_dirs_norm
#     return result * valid  # zero out tokens where mahal_dirs is zero


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
        return mahal_dirs.to(original_dtype)

        # return _proj_to_mahal_dirs(centered, mahal_dirs).to(original_dtype)


class BatchedSVDCollapser:
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
        eig_vec_T = self.eig_vec.mT
        correction = pt._grouped_mm(proj_diff, eig_vec_T, offs=offsets)
        mahal_dirs = centered - correction

        assert mahal_dirs.shape == vecs_sorted.shape
        return mahal_dirs.to(original_dtype)

        # result = _proj_to_mahal_dirs(centered, mahal_dirs)
        # assert result.shape == vecs_sorted.shape
        # return result.to(original_dtype)


# #  it is much slower than BatchedInvSmallCovCollapser, and unalernign trajectory is worse
# class BatchedSmallSVDCollapser:
#     """Like BatchedInvSmallCovCollapser but uses SVD on the small k×k matrix instead of linalg.inv."""

#     P: pt.Tensor        # (E, D, k)
#     old_P: pt.Tensor    # (E, D, k)
#     cov_P: pt.Tensor    # (E, D, k) accumulated X.T @ X @ P
#     inv_cov: pt.Tensor  # (E, k, k) = V @ diag(1/eig_val_normalized) @ V.T

#     def __init__(self, PCs_to_use: int, num_experts: int):
#         self.PCs_to_use = PCs_to_use
#         self.num_experts = num_experts
#         self.P_is_ready = False

#     def add_vecs(self, vecs: pt.Tensor, offsets: pt.Tensor):
#         vecs = vecs.float()
#         if not hasattr(self, "P"):
#             E, D, k = self.num_experts, vecs.shape[1], self.PCs_to_use
#             _rand = pt.randn(E, D, k, dtype=pt.float32, device=vecs.device)
#             self.P = pt.linalg.qr(_rand).Q  # (E, D, k)
#             self.cov_P = pt.zeros_like(self.P)

#         Y = pt._grouped_mm(vecs, self.P, offs=offsets)  # (S, k)
#         self.cov_P += pt._grouped_mm(vecs.mT.contiguous(), Y, offs=offsets)  # (E, D, k)

#         if self.P_is_ready:
#             self.small_second_moment += pt._grouped_mm(Y.mT.contiguous(), Y, offs=offsets)  # (E, k, k)

#     def fit(self):
#         if hasattr(self, "small_second_moment"):
#             S, V = pt.linalg.eigh(self.small_second_moment)  # S: (E, k) ascending, V: (E, k, k)
#             eig_val_norm = S / S.amin(dim=-1, keepdim=True)  # (E, k), normalized so min=1
#             # inv_cov = V @ diag(1/eig_val_norm) @ V.T  (like inv(small_cov) * eigvals_min)
#             self.inv_cov = (V / eig_val_norm.unsqueeze(-2)) @ V.mT.contiguous()  # (E, k, k)
#             self.old_P = self.P

#         self.P = pt.linalg.qr(self.cov_P).Q  # (E, D, k)
#         self.cov_P = pt.zeros_like(self.P)
#         self.small_second_moment = pt.zeros(self.num_experts, self.PCs_to_use, self.PCs_to_use, dtype=pt.float32, device=self.P.device)
#         self.P_is_ready = True

#     def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
#         """Batched collapse. vecs_sorted: (S, D), offsets: (E,)."""
#         original_dtype = vecs_sorted.dtype
#         vecs_sorted = vecs_sorted.float()

#         x_proj = pt._grouped_mm(vecs_sorted, self.old_P, offs=offsets)  # (S, k)
#         correction_proj = x_proj - pt._grouped_mm(x_proj, self.inv_cov, offs=offsets)  # (S, k)
#         correction = pt._grouped_mm(correction_proj, self.old_P.mT.contiguous(), offs=offsets)  # (S, D)
#         mahal_dirs = vecs_sorted - correction

#         result = _proj_to_mahal_dirs(vecs_sorted, mahal_dirs)
#         assert result.shape == vecs_sorted.shape
#         return result.to(original_dtype)


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
