import torch as pt
from torch_incremental_pca import IncrementalPCA
from welford_torch import OnlineCovariance


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


class IncrementalPCACollapser:
    def __init__(self, PCs_to_use: int, device: str):
        self.PCs_to_use = PCs_to_use
        self.device = device
        self.accumulator = None
        self.ipca = IncrementalPCA(n_components=PCs_to_use, gram=True)

    def add_vecs(self, vecs):
        """In addition to partial_fit, it fill also accumulate the vecs if needed"""
        if self.accumulator is not None:
            vecs = pt.cat([self.accumulator, vecs])
            self.accumulator = None

        if vecs.shape[0] < self.ipca.n_components:  # too few vecs, so accumulate
            self.accumulator = vecs
        else:  # enough vecs
            vecs = vecs.clone()  # ipca.partial_fit may modify the input in-place
            self.ipca.partial_fit(vecs)

    def process_saved_vecs(self):
        pass  # components are updated online in partial_fit

    def collapse(self, vecs):
        eig_vec = self.ipca.components_.T  # (n_features, n_components)
        eig_val = self.ipca.explained_variance_  # (n_components,)
        centered = vecs - self.ipca.mean_
        assert centered.dtype == pt.float32
        mahal_dirs = _get_mahal_dirs(centered, eig_val, eig_vec)
        return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)


class CovStoringCollapser:
    mean: pt.Tensor
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, PCs_to_use: int, device: str):
        self.PCs_to_use = PCs_to_use
        self.device = device
        self._reset_vecs()

    def _reset_vecs(self):
        # Reset online covariance for next epoch
        self.online_cov = OnlineCovariance(device=self.device, dtype=pt.bfloat16)

    def add_vecs(self, vecs):
        self.online_cov.add_all(vecs)

    def process_saved_vecs(self):
        # Extract distribution stats from online covariance
        self.online_cov.to_inplace(dtype=pt.float32)
        self.mean = self.online_cov.mean
        cov = self.online_cov.cov
        _, S, V = pt.svd_lowrank(cov, q=self.PCs_to_use)
        self.eig_val = S  # top-k eigenvalues (largest first)
        self.eig_vec = V  # (D, k)
        self._reset_vecs()

    def collapse(self, vecs):
        centered = vecs - self.mean
        mahal_dirs = _get_mahal_dirs(centered, self.eig_val, self.eig_vec)
        return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)
