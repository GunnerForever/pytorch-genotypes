import torch


def dosage_to_hard_call(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Converts a matrix of dosages into hard calls.

    Given the genotype of an individual at a specific variante G ~ Binom(2p),
    E(G) = 2p = dosage.

    With this definition, the most probable genotype assignment can be obtained
    by using 2/3 and 4/3.

    This can be seen by looking at the genotype probabilities:

    P(G = AA) = d ** 2 / 4
    P(G = AB) = d - d ** 2 / 2
    P(G = BB) = 1 - d + d ** 2 / 4

    And taking the maximum probability over d in [0, 2].

    """

    out = torch.ones_like(matrix, dtype=torch.int)

    out[matrix <= 2/3] = 0
    out[matrix >= 4/3] = 2

    return out
