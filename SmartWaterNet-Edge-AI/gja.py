
import time
import numpy as np

from utils import SEED


class GekkoJaponicusAlgorithm:
    """Population-based metaheuristic (gecko perch / move / regenerate)."""

    def __init__(self, n_nodes, fitness_fn, pop_size=30, max_iter=80,
                 w_coverage=1.0, w_redundancy=0.05, w_count=0.02, seed=SEED):
        self.n = n_nodes
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_cov = w_coverage
        self.w_red = w_redundancy
        self.w_cnt = w_count
        self.rng = np.random.default_rng(seed)

    def _binarize(self, pos):
        return (1 / (1 + np.exp(-pos)) > 0.5).astype(int)

    def _fitness(self, mask):
        if mask.sum() == 0:
            return -1e9
        cov, red = self.fitness_fn(mask)
        return self.w_cov * cov - self.w_red * red - self.w_cnt * mask.sum()

    def optimize(self):
        pos = self.rng.normal(0, 1, (self.pop_size, self.n))
        best_mask, best_fit = None, -1e9
        for it in range(self.max_iter):
            masks = np.array([self._binarize(p) for p in pos])
            fits = np.array([self._fitness(m) for m in masks])
            i = int(np.argmax(fits))
            if fits[i] > best_fit:
                best_fit, best_mask = fits[i], masks[i].copy()

            r1 = self.rng.random((self.pop_size, self.n))
            r2 = self.rng.random((self.pop_size, self.n))
            noise = self.rng.normal(0, 0.5 * (1 - it / self.max_iter),
                                    (self.pop_size, self.n))
            pos = pos + r1 * (best_mask - pos) + r2 * noise

            worst = np.argsort(fits)[:max(1, self.pop_size // 10)]
            pos[worst] = self.rng.normal(0, 1, (len(worst), self.n))
        return best_mask.astype(int), best_fit


def coverage_redundancy_fitness(X_nodes, threshold=0.95, min_keep=4):
    def _fn(mask):
        idx = np.where(mask == 1)[0]
        if len(idx) < 2:
            return 0.0, 1.0
        sub = X_nodes[:, idx]
        corr = np.nan_to_num(np.corrcoef(sub, rowvar=False))
        off = np.abs(corr - np.eye(len(idx)))
        coverage = float(np.mean(np.max(off, axis=1) < threshold))
        redundancy = float(np.mean(off))
        shortfall = max(0, min_keep - len(idx)) / min_keep
        return coverage - shortfall, redundancy
    return _fn


def run_gja_sensor_selection(X_nodes, node_names, min_keep=4,
                             pop_size=30, max_iter=80):
    print("\n===== Gekko Japonicus Algorithm - Sensor Node Selection =====")
    print(f"Candidates: {node_names}")
    fit_fn = coverage_redundancy_fitness(X_nodes, threshold=0.95, min_keep=min_keep)
    gja = GekkoJaponicusAlgorithm(n_nodes=X_nodes.shape[1], fitness_fn=fit_fn,
                                  pop_size=pop_size, max_iter=max_iter)
    t0 = time.perf_counter()
    mask, score = gja.optimize()
    dt = time.perf_counter() - t0
    selected = [node_names[i] for i in np.where(mask == 1)[0]]
    print(f"Selected {len(selected)} / {len(node_names)} nodes -> {selected}")
    print(f"Best fitness: {score:.4f}  |  GJA runtime: {dt*1000:.1f} ms")
    return mask, selected, score