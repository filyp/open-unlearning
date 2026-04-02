import torch as pt

# from torch_incremental_pca import IncrementalPCA
from trainer.unlearn.repselect.online_covariance import (
    BatchedOnlineCovariance,
    OnlineCovariance,
)
# todo uninstall welford_torch


def _proj_to_mahal_dirs(centered, mahal_dirs):
    mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
    proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    return proj_strenghts * mahal_dirs_norm


class CovCollapser:
    mean: pt.Tensor
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use
        self.online_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_data = False

    def add_vecs(self, vecs):
        self.online_cov.add_all(vecs)
        self._has_data = True

    def process_saved_vecs(self):
        # if not self._has_data:  # in case an expert was never selected
        #     return
        assert self._has_data, "No data to process"
        # Extract distribution stats from online covariance
        self.mean = self.online_cov.mean.float()
        cov = self.online_cov.cov().float()

        # _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use, niter=0)
        # adapted from svd_lowrank, but niter=0, and provides hot-start (eig_vec)
        init = (
            self.eig_vec  # provides hot-start, instead of random initialization
            if hasattr(self, "eig_vec")
            else pt.randn(
                cov.shape[0], self.PCs_to_use, dtype=cov.dtype, device=cov.device
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
        self._has_data = False

    def collapse(self, vecs):
        centered = vecs - self.mean

        # get Mahalanobis directions
        projected = centered @ self.eig_vec  # (N, D)
        proj_diff = projected - projected / self.eig_val  # assumes eig_val.min() == 1
        mahal_dirs = centered - proj_diff @ self.eig_vec.T

        return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)


class BatchedCovCollapser:
    """Single BatchedOnlineCovariance for all experts; exposes batched collapse via _grouped_mm."""

    mean: pt.Tensor  # (E, D)
    eig_val: pt.Tensor  # (E, k)
    eig_vec: pt.Tensor  # (E, D, k)

    def __init__(self, PCs_to_use: int, num_experts: int):
        self.PCs_to_use = PCs_to_use
        self.num_experts = num_experts
        self.online_cov = BatchedOnlineCovariance(self.num_experts)

    def add_vecs(self, vecs: pt.Tensor, offsets: pt.Tensor, num_experts: int):
        self.online_cov.add_all(vecs, offsets, num_experts)

    def process_saved_vecs(self):
        device = self.online_cov.device
        D = self.online_cov.D
        k = self.PCs_to_use
        init = (
            self.eig_vec
            if hasattr(self, "eig_vec")
            else pt.randn(self.num_experts, D, k, device=device, dtype=pt.float32)
        )
        self.mean = self.online_cov.mean.float()
        self.eig_val = pt.zeros(self.num_experts, k, device=device, dtype=pt.float32)
        self.eig_vec = pt.zeros(self.num_experts, D, k, device=device, dtype=pt.float32)

        for e in range(self.num_experts):
            cov_e = self.online_cov.cov(e).float()

            Q = pt.linalg.qr(cov_e @ init[e]).Q
            B = Q.mT @ cov_e
            _, S, Vh = pt.linalg.svd(B, full_matrices=False)
            V = Vh.mT

            self.eig_val[e] = S / S.min()
            self.eig_vec[e] = V

        # reset online covariance for next epoch
        self.online_cov = BatchedOnlineCovariance(self.num_experts)

    def collapse(self, vecs_sorted: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
        """Batched collapse using _grouped_mm. vecs_sorted: (S, D), offsets: (E,)."""
        dtype = vecs_sorted.dtype
        S = vecs_sorted.shape[0]
        device = vecs_sorted.device

        # Expand per-expert mean to per-token for centering
        expert_ids = pt.bucketize(pt.arange(S, device=device), offsets)
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
        return result.to(dtype)


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


#         _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use, niter=0)
#         self.eig_val = S  # top-k eigenvalues (largest first)
#         self.eig_vec = V  # (D, k)
#         self._reset_vecs()
