import logging
import math

from trainer.unlearn.repselect_simple import RepSelectSimple

logger = logging.getLogger(__name__)

KL_LOW_RATIO = 0.95
BRACKET_FACTOR = 8.0
MAX_ITERS = 30


class RepSelectAdaptive(RepSelectSimple):
    """Adaptive variant of RepSelectSimple.

    Replaces the fixed `num_train_epochs * lr` loop with an exponential-bracket
    + log-bisect search on the cumulative step size alpha, targeting the unique
    KL evaluator that has `disr_budget` set. Lands KL in
    [KL_LOW_RATIO * disr_budget, disr_budget] in O(log) eval calls.

    Each step recomputes w from a stored original: w = w0 - filtered_grad * alpha.
    This avoids accumulating bf16 rounding error across the back-and-forth deltas
    of the bisect (which would otherwise leave w at a stale state and stall the
    search above kl_high).
    """

    def _step_to(self, alpha):
        self.iters += 1
        assert self.iters < MAX_ITERS
        for w in self.base_trainable_params:
            w.data = w.original - w.filtered_grad * alpha
        res = self.evaluator.evaluate(model=self.model, trainer=self)
        return res[f"{self.evaluator.dataset_name}_kl"]

    def _apply_unlearn_loop(self):
        # initial full eval (also primes the KL evaluator's cached hidden states)
        self.evaluate()

        candidates = [
            ev
            for ev in self.evaluators.values()
            if getattr(ev, "disr_budget", None) is not None
        ]
        assert len(candidates) == 1, "expected exactly one disr_budget evaluator"
        self.evaluator = candidates[0]
        kl_high = self.evaluator.disr_budget
        kl_low = KL_LOW_RATIO * kl_high

        # snapshot originals (same dtype as weight) before any modification
        for w in self.base_trainable_params:
            w.original = w.data.clone()

        self.iters = 0
        alpha = float(self.args.learning_rate)
        kl = self._step_to(alpha)
        logger.info(f"adaptive init: alpha={alpha:.4g} kl={kl:.4g}")

        # Phase 1: exponential bracket
        if kl > kl_high:
            alpha_hi = alpha
            while kl > kl_high:
                alpha /= BRACKET_FACTOR
                kl = self._step_to(alpha)
                logger.info(f"adaptive shrink: alpha={alpha:.4g} kl={kl:.4g}")
            alpha_lo = alpha
        else:
            alpha_lo = alpha
            while kl < kl_low:
                alpha *= BRACKET_FACTOR
                kl = self._step_to(alpha)
                logger.info(f"adaptive grow: alpha={alpha:.4g} kl={kl:.4g}")
            alpha_hi = alpha

        # Phase 2: log-bisect until kl in [kl_low, kl_high]
        while not (kl_low <= kl <= kl_high):
            alpha = math.sqrt(alpha_lo * alpha_hi)
            kl = self._step_to(alpha)
            logger.info(
                f"adaptive bisect: [{alpha_lo:.4g}, {alpha_hi:.4g}] "
                f"mid={alpha:.4g} kl={kl:.4g}"
            )
            if kl > kl_high:
                alpha_hi = alpha
            else:
                alpha_lo = alpha

        logger.info(f"adaptive final: alpha={alpha:.4g} kl={kl:.4g}")
        self.state.epoch = 1
        # final full eval at the chosen alpha
        self.evaluate()
