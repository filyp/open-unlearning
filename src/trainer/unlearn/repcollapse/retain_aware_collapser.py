"""Retain-aware CovCollapser: PCA directions ranked by forget/retain ratio."""
import torch as pt
from trainer.unlearn.repcollapse.collapsers import CovCollapser, _proj_to_mahal_dirs
from trainer.unlearn.repcollapse.online_covariance import OnlineCovariance


class RetainAwareCovCollapser(CovCollapser):
    """Drop-in replacement for CovCollapser.

    Same PCA directions as CovCollapser, but eigenvalues are reweighted
    by the forget/retain variance ratio.  Directions that are selective
    (high forget, low retain variance) are preserved more; directions that
    are shared (similar variance in both) are suppressed more.
    """

    def __init__(self, PCs_to_use: int, reg_eps: float = 1e-4):
        super().__init__(PCs_to_use)
        self.reg_eps = reg_eps
        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_retain_data = False

    def add_retain_vecs(self, vecs):
        self.retain_cov.add_all(vecs)
        self._has_retain_data = True

    def _reset_vecs(self):
        super()._reset_vecs()
        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_retain_data = False

    def process_saved_vecs(self):
        """Compute PCA directions, then reweight by forget/retain ratio."""
        # Save retain cov before super() resets it
        has_retain = self._has_retain_data
        if has_retain:
            Sigma_r = self.retain_cov.cov().to(pt.float32)

        # Standard PCA (same as CovCollapser) — this calls _reset_vecs()
        super().process_saved_vecs()

        if not has_retain:
            return  # fall back to standard PCA ranking

        # Compute retain variance along each PCA direction
        V = self.eig_vec  # (D, k) — PCA directions from forget data

        # retain_var_i = V_i^T @ Sigma_r @ V_i for each direction i
        retain_var = (V.T @ Sigma_r @ V).diagonal().clamp(min=self.reg_eps)
        forget_var = self.eig_val * self.eig_val.min()  # unnormalize (approx)

        # Log the structure for analysis
        import logging
        logger = logging.getLogger(__name__)
        ratio = self.eig_val / retain_var
        # Check ranking: does ratio ordering match eig_val ordering?
        pca_rank = self.eig_val.argsort(descending=True)
        ratio_rank = ratio.argsort(descending=True)
        rank_match = (pca_rank == ratio_rank).float().mean().item()
        # Spearman correlation of rankings
        n = len(self.eig_val)
        d = (pca_rank.float() - ratio_rank.float())
        spearman = 1 - 6 * (d ** 2).sum().item() / (n * (n ** 2 - 1))

        logger.info(
            f"RetainAware (dim={V.shape[0]}): "
            f"rank_match={rank_match:.3f}, spearman={spearman:.4f}, "
            f"retain_var top5={[f'{v:.2f}' for v in retain_var[:5].tolist()]}, "
            f"retain_var bot5={[f'{v:.2f}' for v in retain_var[-5:].tolist()]}, "
            f"retain_var std/mean={retain_var.std().item()/retain_var.mean().item():.3f}"
        )

        # Re-normalize so min = 1 (same convention as CovCollapser)
        self.eig_val = ratio / ratio.min()

        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_retain_data = False
