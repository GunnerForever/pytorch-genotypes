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

        if self.use_standardized_genotype:
            if geno_std is None:
                raise RuntimeError(f"No standardized genotypes provided in "
                                   f"the dataset. Batch: {batch}")
            geno_logits = self.forward(geno_std)
        else:
            geno_logits = self.forward(geno_dosage)

        geno_p = expit(geno_logits)
        reconstruction_dosage = 2 * geno_p

        reconstruction_mse = F.mse_loss(reconstruction_dosage, geno_dosage)

        p_geno2 = geno_p ** 2
        p_geno1 = 2 * geno_p * (1 - geno_p) + 0.5
        p_geno0 = (1 - geno_p) ** 2

        smax_0 = torch.exp(p_geno0)
        smax_1 = torch.exp(p_geno1)
        smax_2 = torch.exp(p_geno2)

        smax_c = smax_0 + smax_1 + smax_2

        def clamp(x):
            return torch.clamp(x, min=-10)

        preds_three_classes = -clamp(torch.log(torch.stack((
            smax_0 / smax_c,
            smax_1 / smax_c,
            smax_2 / smax_c
        ), dim=2)))

        if self.softmax_weights is not None:
            preds_three_classes *= self.softmax_weights

        # Index to get the target likelihood.
        hard_calls = dosage_to_hard_call(geno_dosage)
        likelihoods = torch.gather(
            preds_three_classes,
            dim=2,
            index=hard_calls.unsqueeze(-1)
        )

        mean_like = likelihoods.mean()
        if torch.isnan(mean_like):
            print("==> LOSS IS NAN!! <==")

        if log:
            reconstruction_hard_calls = dosage_to_hard_call(
                reconstruction_dosage
            )

            accurate_reconstruction = (
                reconstruction_hard_calls == hard_calls
            ).to(torch.float32)

            reconstruction_acc = accurate_reconstruction.mean()

            reconstruction_nll = (
                -geno_dosage * clamp(torch.log(geno_p))
                - (2 - geno_dosage) * clamp(torch.log(1 - geno_p))
            ).mean()

            # This may be computationally intensive but can be useful to debug.
            homo_min = hard_calls == 0
            het = hard_calls == 1
            homo_maj = hard_calls == 2

            self.log(f"{log_prefix}homo_min_acc",
                     accurate_reconstruction[homo_min].mean())
            self.log(f"{log_prefix}het_acc",
                     accurate_reconstruction[het].mean())
            self.log(f"{log_prefix}homo_maj_acc",
                     accurate_reconstruction[homo_maj].mean())

            self.log(f"{log_prefix}mean_like", mean_like)

            self.log(f"{log_prefix}reconstruction_acc", reconstruction_acc)
            self.log(f"{log_prefix}reconstruction_nll", reconstruction_nll)
            self.log(f"{log_prefix}reconstruction_mse", reconstruction_mse)

        return mean_like

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

    def _get_x_from_batch(self, batch):
        if len(batch) == 2:
            geno_dosage, geno_std = batch
        else:
            geno_dosage = batch[0]
            geno_std = None

        if self.use_standardized_genotype:
            if geno_std is None:
                raise ValueError(
                    "Can't train GenotypeAutoencoder on standardized genotypes"
                    "unless supported by the dataset."
                )

            return geno_std.to(torch.float32)

        return geno_dosage.to(torch.float32)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
