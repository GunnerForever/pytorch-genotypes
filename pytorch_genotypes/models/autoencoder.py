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
from torch.special import expit
import pytorch_lightning as pl


from ..utils import dosage_to_hard_call


class GenotypeAutoencoder(pl.LightningModule):
    """Genotype autoencoder that predicts individual minor allele frequency.

    The output of the decoder is a vector of probabilities (actually logits)
    for every sample to be interpreted as a minor allele probability.

    This model is trained by negative log likelihood and will convert the input
    from dosages to genotype calls if needed.

    When generating reconstructions, we take 2 * individual mutation
    probability.

    """
    def __init__(
        self,
        encoder: Optional[pl.LightningModule] = None,
        decoder: Optional[pl.LightningModule] = None,
        lr=1e-3,
        weight_decay=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder

    def _step(self, geno_dosage, batch_idx, log, log_prefix=""):
        geno_logits = self.forward(geno_dosage)

        geno_p = expit(geno_logits)
        reconstruction_dosage = 2 * geno_p

        reconstruction_nll = (
            -geno_dosage * torch.clamp(torch.log(geno_p), min=-100)
            - (2 - geno_dosage) * torch.clamp(torch.log(1 - geno_p), min=-100)
        ).mean()

        if log:
            hard_calls = dosage_to_hard_call(geno_dosage)
            reconstruction_acc = (
                dosage_to_hard_call(reconstruction_dosage) ==
                hard_calls
            ).to(torch.float32).mean(dim=1).mean()

            self.log(f"{log_prefix}reconstruction_acc", reconstruction_acc)
            self.log(f"{log_prefix}reconstruction_nll", reconstruction_nll)

        return reconstruction_nll

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
        dosage = self._get_dosage_from_batch(batch)
        return self._step(dosage, batch_idx, log=True, log_prefix="train_")

    def validation_step(self, batch, batch_idx):
        dosage = self._get_dosage_from_batch(batch)
        return self._step(dosage, batch_idx, log=True, log_prefix="val_")

    def test_step(self, batch, batch_idx):
        dosage = self._get_dosage_from_batch(batch)
        return self._step(dosage, batch_idx, log=True, log_prefix="test_")

    @staticmethod
    def _get_dosage_from_batch(batch):
        if len(batch) == 2:
            geno_dosage, _ = batch
        else:
            geno_dosage = batch[0]

        return geno_dosage.to(torch.float32)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
