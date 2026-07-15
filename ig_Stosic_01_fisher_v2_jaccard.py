from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
Uslovna raspodela: 
Jaccard top-50 → next → p_cond → Fisher; skor vs global p; next na kraju.


IG korak 1 v2 — Fisher na uslovnoj raspodeli (ne globalna frekvencija).

p(y | last) ≈ empirija next posle top-K Jaccard-sličnih kola last-u.
  rate_i = P(i ∈ next | similar last)
  p_i = rate_i / sum(rate)     (simplex uslovne mase)
  g_ii = 1 / p_i

Skor: (p_cond − p_global) · √g_cond  — bežim od čiste frekvencije.
next: jedna kombinacija. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
K_SIM = 50
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def jaccard(a, b) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def conditional_next_rates(draws: np.ndarray, query: np.ndarray, k: int = K_SIM) -> tuple[np.ndarray, np.ndarray]:
    """rate_i = udeo top-k sličnih čiji next sadrži broj i; vraća (rate, next_members)."""
    sims = [(i, jaccard(draws[i], query)) for i in range(len(draws) - 1)]
    sims.sort(key=lambda t: (-t[1], t[0]))
    top = [i for i, _ in sims[:k]]
    nexts = draws[[i + 1 for i in top]]
    rate = np.zeros(FRONT_N, dtype=float)
    for d in nexts:
        for x in d.tolist():
            rate[int(x) - 1] += 1.0
    rate /= max(len(top), 1)
    return rate, nexts


def rates_to_simplex(rate: np.ndarray) -> np.ndarray:
    """Laplace + normalizacija → uslovna p na simplexu."""
    mass = rate + 1e-6
    return mass / mass.sum()


def fisher_diagonal(p: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(p, 1e-18, None)


def number_scores(p_cond: np.ndarray, p_glob: np.ndarray, g: np.ndarray) -> dict[int, float]:
    """Uslovni excess vs global, Fisher težina."""
    return {
        i + 1: float((p_cond[i] - p_glob[i]) * np.sqrt(g[i]))
        for i in range(FRONT_N)
    }


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
) -> float:
    nums = sorted(combo)
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(
    p_cond: np.ndarray,
    p_glob: np.ndarray,
    g: np.ndarray,
    members: np.ndarray,
) -> list[int]:
    score = number_scores(p_cond, p_glob, g)
    ranked = sorted(score, key=lambda n: (-score[n], n))
    target_sum = float(members.sum(axis=1).mean())
    pos_means = [float(members[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in members]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_01_v2(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    p_glob = global_p(draws)
    rate, members = conditional_next_rates(draws, draws[-1], k=K_SIM)
    p_cond = rates_to_simplex(rate)
    g = fisher_diagonal(p_cond)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | K_SIM={K_SIM} | ig_01_v2 Fisher uslovno")
    print(f"last: {draws[-1].tolist()}")
    print()

    anisotropy = float(g.max() / g.min()) if g.min() > 0 else float("inf")
    print("=== uslovna p + Fisher ===")
    print(
        {
            "sum_p_cond": round(float(p_cond.sum()), 6),
            "rate_max": float(rate.max()),
            "p_cond_min": float(p_cond.min()),
            "p_cond_max": float(p_cond.max()),
            "g_min": float(g.min()),
            "g_max": float(g.max()),
            "anisotropy": round(anisotropy, 4),
        }
    )
    print()

    score = number_scores(p_cond, p_glob, g)
    ranked_score = sorted(
        (
            (n, float(rate[n - 1]), float(p_cond[n - 1]), float(score[n]))
            for n in range(1, FRONT_N + 1)
        ),
        key=lambda t: (-t[3], t[0]),
    )
    print("=== top12 po (p_cond − p_glob)·√g ===")
    print(
        [
            (n, round(r, 3), round(pc, 5), round(sc, 5))
            for n, r, pc, sc in ranked_score[:12]
        ]
    )
    print()

    combo = predict_next(p_cond, p_glob, g, members)
    print("=== next (ig_01_v2 uslovni Fisher) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_01_v2()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | K_SIM=50 | ig_01_v2 Fisher uslovno
last: [4, 5, 6, 11, 12, 18, 28]

=== uslovna p + Fisher ===
{'sum_p_cond': 1.0, 'rate_max': 0.34, 'p_cond_min': 0.00571439673407534, 'p_cond_max': 0.04857130081703831, 'g_min': 20.588289446207508, 'g_max': 174.99660008499785, 'anisotropy': 8.4998}

=== top12 po (p_cond − p_glob)·√g ===
[(30, 0.32, 0.04571, 0.10044), (26, 0.34, 0.04857, 0.09897), (23, 0.28, 0.04, 0.06052), (8, 0.28, 0.04, 0.05945), (38, 0.26, 0.03714, 0.05802), (18, 0.24, 0.03429, 0.04828), (5, 0.24, 0.03429, 0.04712), (35, 0.24, 0.03429, 0.04496), (7, 0.24, 0.03429, 0.04413), (13, 0.22, 0.03143, 0.0331), (20, 0.2, 0.02857, 0.02835), (14, 0.2, 0.02857, 0.02145)]

=== next (ig_01_v2 uslovni Fisher) ===
next: [5, 8, 13, 23, 26, 30, 38]
"""
