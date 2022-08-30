"""
Implementations for some commonly used neural network architectures.

All models are expected to be trained with (Optional):

- Genotype dosage
- Genotype standardized
- Exogenous variables (covariates)
- Endogenous variable(s) (targets for supervised models)

"""


import torch
import torch.nn.functional as F
from torch.special import expit
import pytorch_lightning as pl


from ..utils import dosage_to_hard_call


class GenotypeProbabilisticAutoencoder(pl.LightningModule):
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
        lr,
        weight_decay,
        encoder: pl.LightningModule,
        decoder: pl.LightningModule,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def _step(self, batch, batch_idx, log, log_prefix=""):
        geno_dosage = batch
        hard_calls = dosage_to_hard_call(geno_dosage)

        geno_logits = self.forward(geno_dosage)
        reconstruction_dosage = 2 * expit(geno_logits)

        reconstruction_mse = F.mse_loss(geno_dosage, reconstruction_dosage)

        if log:
            reconstruction_acc = (
                dosage_to_hard_call(reconstruction_dosage) ==
                hard_calls
            ).to(float).mean(dim=1).mean()

            self.log(f"{log_prefix}reconstruction_acc", reconstruction_acc)
            self.log(f"{log_prefix}reconstruction_mse", reconstruction_mse)

        return reconstruction_mse

    def encode(self, geno_std):
        """Return the latent code for a given sample."""
        return self.encoder(geno_std)

    def decode(self, zs):
        """Return the reconstructed mutations logits from a code."""
        return self.decoder(zs)

    def forward(self, geno_std):
        """A forward pass through the autoencoder returning mutation logits."""
        return self.decode(self.encode(geno_std))

    def training_step(self, batch, batch_idx):
        batch = batch.to(torch.float32)
        return self._step(batch, batch_idx, log=True)

    def test_step(self, batch, batch_idx):
        batch = batch.to(torch.float32)
        return self._step(batch, batch_idx, log=True, log_prefix="test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
