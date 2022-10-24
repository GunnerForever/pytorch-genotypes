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

"""
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("1kg_common_norel.afreq", sep="\t")
import torch
mafs = torch.from_numpy(df.ALT_FREQS.values)

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

freqs = geno_freqs_from_maf(mafs)
entropies = variant_entropy_from_geno_freqs(freqs)

max_ent = torch.max(entropies)
clipped_inv_freqs = torch.clip(1 / freqs, max=max_ent)
weights = clipped_inv_freqs * entropies[:, None]

small = weights.numpy()[:5000, :]
fig, axes = plt.subplots(3, 1, figsize=(5, 16))
axes[0].hist(small[:, 0], bins=50)
axes[0].set_title("Weights AA")
axes[1].hist(small[:, 1], bins=50, label="Weights AB")
axes[1].set_title("Weights AB")
axes[2].hist(small[:, 2], bins=50, label="Weights BB")
axes[2].set_title("Weights BB")

"""
