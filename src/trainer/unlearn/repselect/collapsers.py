import torch as pt
# from torch_incremental_pca import IncrementalPCA
from trainer.unlearn.repselect.online_covariance import OnlineCovariance
# todo uninstall welford_torch


def _get_mahal_dirs(centered, eig_val, eig_vec):
    # Compute Mahalanobis directions using eigendecomposition
    projected = centered @ eig_vec  # (N, D)
    proj_diff = projected - projected / (eig_val / eig_val.min())
    # neg = projected / (eig_val / eig_val.min())
    # neg[:, -24:] = 0
    # proj_diff = projected - neg
    return centered - proj_diff @ eig_vec.T


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
        self._reset_vecs()

    def _reset_vecs(self):
        # Reset online covariance for next epoch
        self.online_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_data = False

    def add_vecs(self, vecs):
        self.online_cov.add_all(vecs)
        self._has_data = True

    def process_saved_vecs(self):
        if not self._has_data:  # in case an expert was never selected
            return
        # Extract distribution stats from online covariance
        self.mean = self.online_cov.mean.to(pt.float32)
        cov = self.online_cov.cov().to(pt.float32)
        _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use)
        self.eig_val = S  # top-k eigenvalues (largest first)
        self.eig_vec = V  # (D, k)
        self._reset_vecs()

    def collapse(self, vecs):
        centered = vecs - self.mean
        mahal_dirs = _get_mahal_dirs(centered, self.eig_val, self.eig_vec)
        return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)


class BatchedCovCollapser:
    """Wraps E individual CovStoringCollapsers, exposes batched collapse via _grouped_mm."""

    mean: pt.Tensor     # (E, D)
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
        # Stack per-expert stats into batched tensors; keep previous for experts with no new data
        means, eig_vals, eig_vecs = [], [], []
        for i, c in enumerate(self.collapsers):
            assert hasattr(c, "mean"), f"Expert {i} has no stats — was it never routed any tokens?"
            means.append(c.mean)
            eig_vals.append(c.eig_val)
            eig_vecs.append(c.eig_vec)
        self.mean = pt.stack(means)       # (E, D)
        self.eig_val = pt.stack(eig_vals) # (E, k)
        self.eig_vec = pt.stack(eig_vecs) # (E, D, k)

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
        eig_val_tok = self.eig_val[expert_ids]  # (S, k) — small
        eig_val_tok = eig_val_tok / eig_val_tok.min(dim=1, keepdim=True).values
        proj_diff = projected - projected / eig_val_tok

        # Back-project: (S, k) × (E, k, D) → (S, D)
        eig_vec_T = self.eig_vec.transpose(-2, -1).contiguous()
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

