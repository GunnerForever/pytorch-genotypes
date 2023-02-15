"""
Implementations for some commonly used neural network architectures.

All models are expected to be trained with (Optional):

- Genotype dosage
- Genotype standardized
- Exogenous variables (covariates)
- Endogenous variable(s) (targets for supervised models)

"""


from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import expit
import pytorch_lightning as pl


from ..utils import dosage_to_hard_call


class GenotypeAutoencoderAbstract(pl.LightningModule):
    def _step(self, batch, batch_idx, log=True, log_prefix="train_"):
        raise NotImplementedError()

    def encode(self, x):
        """Return the latent code for a given sample."""
        return self.encoder(x)

    def decode(self, zs):
        """Return the reconstructed mutations logits from a code."""
        return self.decoder(zs)

    def forward(self, x):
        """A forward pass through the autoencoder returning mutation logits."""
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log=True, log_prefix="train_")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log=True, log_prefix="val_")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log=True, log_prefix="test_")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


class GenotypeAutoencoder3Classes(GenotypeAutoencoderAbstract):
    """Genotype autoencoder using a three class softmax to represent
    genotypes.

    """
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        use_standardized_genotype: bool = False,
        softmax_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.use_standardized_genotype = use_standardized_genotype
        self.encoder = encoder
        self.decoder = decoder

    def _step(self, batch, batch_idx, log, log_prefix):
        if len(batch) == 2:
            geno_dosage, geno_std = batch
        else:
            geno_dosage = batch[0]
            geno_std = None

        geno_dosage = geno_dosage.to(torch.float32)
        n_mb, n_snps = geno_dosage.shape

        if self.use_standardized_genotype:
            if geno_std is None:
                raise RuntimeError(f"No standardized genotypes provided in "
                                   f"the dataset. Batch: {batch}")
            x = self.forward(geno_std)
        else:
            x = self.forward(geno_dosage)

        x = x.view(n_mb, 3, n_snps)
        hard_calls = dosage_to_hard_call(geno_dosage)
        loss = F.cross_entropy(x, hard_calls)

        if log:
            reconstruction_geno_probs = F.softmax(x, dim=1)
            reconstruction_hard_calls = torch.argmax(
                reconstruction_geno_probs, dim=1
            )

            reconstruction_dosage = torch.einsum(
                "ijk,j->ik",
                reconstruction_geno_probs,
                torch.tensor([0., 1., 2.], device=self.device)
            )

            mse = F.mse_loss(reconstruction_dosage, geno_dosage)

            accurate_reconstruction = (
                reconstruction_hard_calls == hard_calls
            ).to(torch.float32)

            reconstruction_acc = accurate_reconstruction.mean()

            homo_min = hard_calls == 2
            het = hard_calls == 1
            homo_maj = hard_calls == 0

            self.log(f"{log_prefix}homo_min_acc",
                     accurate_reconstruction[homo_min].mean())
            self.log(f"{log_prefix}het_acc",
                     accurate_reconstruction[het].mean())
            self.log(f"{log_prefix}homo_maj_acc",
                     accurate_reconstruction[homo_maj].mean())

            self.log(f"{log_prefix}reconstruction_acc", reconstruction_acc)
            self.log(f"{log_prefix}loss", loss)
            self.log(f"{log_prefix}mse", mse)

        return loss


class GenotypeAutoencoder2Latent(GenotypeAutoencoderAbstract):
    """Genotype autoencoder using two latent variables to represent genotypes.

    z0 is an indicator for heterozygotes.
    z1 is an indicator distinguishing major allele vs minor allele homozygotes.

    The full likelihood for an additively encoded genotype G is:

    P(G=0 | z0, z1) = (1 - z0) * (1 - z1)
    P(G=1 | z0, z1) = z0
    P(G=2 | z0, z1) = (1 - z0) * z1

    The output of the decoder is a vector of probabilities (actually logits)
    for every sample to be interpreted as a minor allele probability.

    """
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        lr=1e-3,
        weight_decay=1e-3,
        use_standardized_genotype: bool = False,
        _save_hyperparameters: bool = True
    ):
        super().__init__()
        if _save_hyperparameters:
            self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.use_standardized_genotype = use_standardized_genotype
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def latents_to_likelihood(z0, z1):
        p_1 = z0
        p_0 = (1 - z0) * (1 - z1)
        p_2 = (1 - z0) * z1

        return torch.dstack((p_0, p_1, p_2))

    def _step(self, batch, batch_idx, log, log_prefix=""):
        if len(batch) == 2:
            geno_dosage, geno_std = batch
        else:
            geno_dosage = batch[0]
            geno_std = None

        geno_dosage = geno_dosage.to(torch.float32)
        n_mb, n_snps = geno_dosage.shape

        if self.use_standardized_genotype:
            if geno_std is None:
                raise RuntimeError(f"No standardized genotypes provided in "
                                   f"the dataset. Batch: {batch}")
            x = self.forward(geno_std)
        else:
            x = self.forward(geno_dosage)

        # dim should be mb x n_snps x 2
        hard_calls = dosage_to_hard_call(geno_dosage)
        x = x.view(n_mb, n_snps, 2)
        x = expit(x)

        likelihood = self.latents_to_likelihood(
            z0=x[:, :, 0],
            z1=x[:, :, 1]
        )
        nll = -torch.log(torch.gather(
            likelihood,
            dim=2,
            index=hard_calls.unsqueeze(-1)
        ))

        reconstruction_dosage = likelihood[:, :, 1] + likelihood[:, :, 2] * 2
        reconstruction_hard_calls = dosage_to_hard_call(
            reconstruction_dosage
        )

        mse = F.mse_loss(reconstruction_dosage, geno_dosage)
        loss = torch.mean(nll)

        if torch.isnan(loss):
            print("==> LOSS IS NAN!! <==")

        if log:
            accurate_reconstruction = (
                reconstruction_hard_calls == hard_calls
            ).to(torch.float32)

            reconstruction_acc = accurate_reconstruction.mean()

            # This may be computationally intensive but can be useful to debug.
            homo_min = hard_calls == 2
            het = hard_calls == 1
            homo_maj = hard_calls == 0

            self.log(f"{log_prefix}homo_min_acc",
                     accurate_reconstruction[homo_min].mean())
            self.log(f"{log_prefix}het_acc",
                     accurate_reconstruction[het].mean())
            self.log(f"{log_prefix}homo_maj_acc",
                     accurate_reconstruction[homo_maj].mean())

            self.log(f"{log_prefix}reconstruction_acc", reconstruction_acc)
            self.log(f"{log_prefix}loss", loss)
            self.log(f"{log_prefix}mse", mse)

        return loss
