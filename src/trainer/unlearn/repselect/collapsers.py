import torch as pt
# from torch_incremental_pca import IncrementalPCA
# from trainer.unlearn.repselect.online_covariance import OnlineCovariance, OnlineCovarianceSimple
# todo uninstall welford_torch


def _proj_to_mahal_dirs(centered, mahal_dirs):
    mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
    proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    return proj_strenghts * mahal_dirs_norm


class CovCollapser:
    def __init__(self, PCs_to_use: int):
        self.PCs_to_use = PCs_to_use
        self.cov = None
        self.P = None
        self.P_is_ready = False

    def add_vecs(self, vecs):
        if self.P is None:
            dim = vecs.shape[1]
            P = pt.randn(dim, self.PCs_to_use, dtype=vecs.dtype, device=vecs.device)
            self.P = pt.linalg.qr(P.float()).Q.bfloat16()
            self.future_P = pt.zeros_like(self.P)

        self.future_P += vecs.mT @ (vecs @ self.P)

        if self.P_is_ready:
            # project vecs into the smaller space
            Y = vecs @ self.P
            if self.cov is None:
                dim = Y.shape[1]
                self.cov = pt.zeros(dim, dim, dtype=pt.bfloat16, device=vecs.device)
            self.cov += Y.mT @ Y

    def process_saved_vecs(self):
        # Extract distribution stats from online covariance
        if self.cov is not None:
            self.inv_cov = pt.linalg.inv(self.cov.float()).bfloat16()
            self.old_P = self.P
            self.cov = None

            # estimate eigvals_min using power iteration
            v = pt.randn(self.PCs_to_use, device=self.inv_cov.device)
            for _ in range(10):
                v = self.inv_cov.float() @ v
                v = v / v.norm()
            self.eigvals_min = 1.0 / (v @ self.inv_cov.float() @ v)

        self.P = pt.linalg.qr(self.future_P.float()).Q.bfloat16()
        self.P_is_ready = True

    def collapse(self, vecs):
        assert vecs.dtype == pt.bfloat16
        vecs = vecs.float()
        inv_cov = self.inv_cov.float()
        old_P = self.old_P.float()

        # get mahalanobis directions
        x_proj = vecs @ old_P
        correction_proj = x_proj - self.eigvals_min * (x_proj @ inv_cov)
        correction = correction_proj @ old_P.T  # lift back
        mahal_dirs = vecs - correction  # only touches the P subspace

        return _proj_to_mahal_dirs(vecs, mahal_dirs).bfloat16()



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

