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
import torch.nn.functional as F
from torch.special import expit
import pytorch_lightning as pl


from ..utils import dosage_to_hard_call


class GenotypeAutoencoder(pl.LightningModule):
    """Genotype autoencoder that predicts individual minor allele frequency.

    The output of the decoder is a vector of probabilities (actually logits)
    for every sample to be interpreted as a minor allele probability.

    For the softmax_weights, if you provide a n_snp x 3 tensor, the penalties
    can be specified by variant and by genotype. If a 3 dimensional tensor is
    provided it will be used as weights for genotypes across all variants.


    """
    def __init__(
        self,
        encoder: Optional[pl.LightningModule] = None,
        decoder: Optional[pl.LightningModule] = None,
        lr=1e-3,
        weight_decay=1e-3,
        use_standardized_genotype: bool = False,
        softmax_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.use_standardized_genotype = use_standardized_genotype
        self.encoder = encoder
        self.decoder = decoder
        self.softmax_weights = softmax_weights

    def _step(self, batch, batch_idx, log, log_prefix=""):
        # Move softmax weights if necessary.
        if self.softmax_weights is not None:
            if self.softmax_weights.device != self.device:
                self.softmax_weights = self.softmax_weights.to(self.device)

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
        # We interpret the two last dimensions as mutation probabilities by
        # strand.
        x = x.view(n_mb, n_snps, 2)
        x = expit(x)
        reconstruction_dosage = (
            # Tring using the first var as an indicator for heterozygotes.
            x[:, :, 0] + (1 - x[:, :, 0]) * 2 * x[:, :, 1]
        )

        reconstruction_hard_calls = dosage_to_hard_call(
            reconstruction_dosage
        )

        hard_calls = dosage_to_hard_call(geno_dosage)
        mse = F.mse_loss(reconstruction_dosage, geno_dosage)

        loss = mse

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

        return loss

    def encode(self, geno_std):
        """Return the latent code for a given sample."""
        return self.encoder(geno_std)

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
