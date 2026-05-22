#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd

# def simulate_Q(n_individuals, k, alpha, rng):
#     """
#     Simulate ancestry proportions from a Dirichlet distribution.
#     Smaller alpha = less admixed; larger alpha = more admixed.
#     """
#     return rng.dirichlet(alpha * np.ones(k), size=n_individuals)


def simulate_Q_admixed(pop_labels, k, alpha, rng, min_pops=2, max_pops=4):
    """
    Simulate admixed ancestry proportions anchored to assigned populations.

    For each assigned population p, randomly choose 2-4 populations including
    p, draw a Dirichlet distribution only over that subset, and leave all other
    populations at zero. The draw is accepted only when p is the maximum
    ancestry component.
    """
    Q = np.zeros((len(pop_labels), k))
    min_pops = max(1, min(min_pops, k))
    max_pops = max(min_pops, min(max_pops, k))

    for i, p in enumerate(pop_labels):
        n_admixed_pops = rng.integers(min_pops, max_pops + 1)
        other_pops = np.setdiff1d(np.arange(k), [p])
        sampled_others = rng.choice(
            other_pops,
            size=n_admixed_pops - 1,
            replace=False,
        )
        pop_subset = np.concatenate([[p], sampled_others])

        while True:
            q_subset = rng.dirichlet(alpha * np.ones(n_admixed_pops))
            if q_subset[0] == q_subset.max():
                Q[i, pop_subset] = q_subset
                break

    return Q


def assign_populations(n_individuals, k):
    """
    Assign individuals evenly to K populations.
    """
    labels = np.repeat(np.arange(k), n_individuals // k)

    if len(labels) < n_individuals:
        extra = np.arange(n_individuals - len(labels))
        labels = np.concatenate([labels, extra])

    return labels


def simulate_Q_discrete(pop_labels, k, admixture=0.02):
    """
    Mostly discrete ancestry, with small background admixture.
    """
    n = len(pop_labels)
    Q = np.full((n, k), admixture / (k - 1))

    for i, p in enumerate(pop_labels):
        Q[i, :] = admixture / (k - 1)
        Q[i, p] = 1.0 - admixture

    return Q


def build_M(
    k,
    n_loci,
    scenario,
    delta,
    informative_fraction,
    direction_bias_fraction,
    rng,
):
    """
    Build population-specific STR deviation matrix M.

    M[k, l] = expected standardized STR length deviation
              for population k at locus l.
    """
    M = np.zeros((k, n_loci))

    n_informative = int(n_loci * informative_fraction)
    n_directional = int(n_loci * direction_bias_fraction)

    loci = np.arange(n_loci)
    rng.shuffle(loci)

    if scenario == "null":
        return M

    if scenario == "symmetric":
        informative_loci = loci[:n_informative]

        for idx, locus in enumerate(informative_loci):
            pop = idx % k
            sign = 1 if idx % 2 == 0 else -1
            M[pop, locus] = sign * delta

    elif scenario == "expansion_only":
        informative_loci = loci[:n_informative]

        for idx, locus in enumerate(informative_loci):
            pop = idx % k
            M[pop, locus] = delta

    elif scenario == "mixed":
        symmetric_loci = loci[:n_informative]
        directional_loci = loci[n_informative:n_informative + n_directional]

        # Symmetric ancestry-informative loci
        for idx, locus in enumerate(symmetric_loci):
            pop = idx % k
            sign = 1 if idx % 2 == 0 else -1
            M[pop, locus] = sign * delta

        # Direction-biased loci
        for idx, locus in enumerate(directional_loci):
            pop = idx % k
            M[pop, locus] = delta

    else:
        raise ValueError(
            "scenario must be one of: null, symmetric, expansion_only, mixed"
        )

    return M


def simulate_STR_matrix(Q, M, sigma, rng):
    """
    Generate standardized STR matrix:
        D = Q M + noise

    Internally, D is samples x STRs. Output writing transposes it to
    STRs x samples for dNMF.py.
    """
    noise = rng.normal(loc=0.0, scale=sigma, size=(Q.shape[0], M.shape[1]))
    D = Q @ M + noise

    return D 


def main():
    parser = argparse.ArgumentParser(
        description="Simulate STR genotype matrices for dNMF testing."
    )

    parser.add_argument("--scenario", default="mixed",
                        choices=["null", "symmetric", "expansion_only", "mixed"])
    parser.add_argument("--n_individuals", type=int, default=500)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_loci", type=int, default=5000)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--min-admixed-pops",
        type=int,
        default=2,
        help="Minimum number of populations with nonzero Q in admixed mode."
    )
    parser.add_argument(
        "--max-admixed-pops",
        type=int,
        default=4,
        help="Maximum number of populations with nonzero Q in admixed mode."
    )
    parser.add_argument("--informative_fraction", type=float, default=0.10)
    parser.add_argument("--direction_bias_fraction", type=float, default=0.05)
    parser.add_argument("--mode", choices=["discrete", "admixed"], default="discrete")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_prefix", default="simulated_STR")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    pop_labels = assign_populations(args.n_individuals, args.k)

    if args.mode == "discrete":
        Q = simulate_Q_discrete(pop_labels, args.k)
    else:
        Q = simulate_Q_admixed(
            pop_labels,
            args.k,
            alpha=args.alpha,
            rng=rng,
            min_pops=args.min_admixed_pops,
            max_pops=args.max_admixed_pops,
        )

    M = build_M(
        k=args.k,
        n_loci=args.n_loci,
        scenario=args.scenario,
        delta=args.delta,
        informative_fraction=args.informative_fraction,
        direction_bias_fraction=args.direction_bias_fraction,
        rng=rng,
    )

    D = simulate_STR_matrix(Q, M, args.sigma, rng)

    sample_ids = [f"ind_{i}" for i in range(args.n_individuals)]
    str_ids = [f"str_{i}" for i in range(args.n_loci)]

    pd.DataFrame(D.T, index=str_ids, columns=sample_ids).to_csv(
        f"{args.out_prefix}.D.tsv", sep="\t"
    )

    pd.DataFrame(Q).to_csv(f"{args.out_prefix}.true_Q.tsv", sep="\t", index=False, header=False)
    pd.DataFrame(M).to_csv(f"{args.out_prefix}.true_M.tsv", sep="\t", index=False, header=False)

    meta = pd.DataFrame({
        "individual": sample_ids,
        "population": [f"pop_{p}" for p in pop_labels],
    })
    meta.to_csv(f"{args.out_prefix}.metadata.tsv", sep="\t", index=False)

    print("Simulation complete.")
    print(f"Scenario: {args.scenario}")
    print(f"Output prefix: {args.out_prefix}")
    print("Generated files:")
    print(f"  {args.out_prefix}.D.tsv")
    # print(f"  {args.out_prefix}.D_pos.tsv")
    # print(f"  {args.out_prefix}.D_neg.tsv")
    print(f"  {args.out_prefix}.true_Q.tsv")
    print(f"  {args.out_prefix}.true_M.tsv")
    print(f"  {args.out_prefix}.metadata.tsv")


if __name__ == "__main__":
    main()
