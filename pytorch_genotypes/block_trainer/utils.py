import torch


def variant_entropy_from_geno_freqs(geno_freqs: torch.Tensor):
    return -torch.sum(
        geno_freqs * torch.clip(torch.log(geno_freqs), min=-10),
        dim=1
    )


def geno_freqs_from_maf(mafs: torch.Tensor):
    p2 = mafs ** 2
    p1 = 2 * mafs * (1 - mafs)
    p0 = (1 - mafs) ** 2

    return torch.vstack((p0, p1, p2)).T
