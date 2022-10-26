"""
Implementation of a dataset that add phenotype data.
"""


from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .core import (BatchDosage, BatchDosageStandardized, BatchFactory,
                   GeneticDataset, GeneticDatasetBackend)

_STANDARDIZE_MAX_SAMPLE = 10_000


BatchExogenous = BatchFactory("BatchExogenous", ("exogenous", ))
BatchEndogenous = BatchFactory("BatchEndogenous", ("endogenous", ))
BatchExogenousEndogenous = BatchFactory(
    "BatchExogenousEndogenous", ("exogenous", "endogenous")
)


def _get_phenotype_batch_factory(
    has_exog: bool,
    has_endog: bool
) -> BatchFactory:
    if has_exog and has_endog:
        return BatchExogenousEndogenous
    elif has_exog and (not has_endog):
        return BatchExogenous
    elif has_endog and (not has_exog):
        return BatchEndogenous
    else:
        raise ValueError()


@dataclass
class StdVariableScaler(object):
    name: str
    mean: float
    std: float

    def scale(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


class PhenotypeGeneticDataset(GeneticDataset):
    """Dataset implementation for genotype-phenotype models.

    Uses an arbitrary backend for the genetic data and a pandas DataFrame to
    hold the correpsonding phenotypes.

    """
    def __init__(
        self,
        backend: GeneticDatasetBackend,
        phenotypes: pd.DataFrame,
        phenotypes_sample_id_column: str = "sample_id",
        exogenous_columns: Optional[List[str]] = None,
        endogenous_columns: Optional[List[str]] = None,
        standardize_columns: Optional[List[str]] = None,
        standardize_genotypes: bool = False
    ):
        super().__init__(backend, genotype_standardization=False)

        if exogenous_columns is None and endogenous_columns is None:
            raise ValueError("Need to provide exogenous_columns or/and "
                             "endogenous_columns.")

        # Check that we can find all the phenotype data.
        expected_cols = {phenotypes_sample_id_column}
        if exogenous_columns is not None:
            expected_cols &= set(exogenous_columns)

        if endogenous_columns is not None:
            expected_cols &= set(endogenous_columns)

        missing_cols = expected_cols - set(phenotypes.columns)
        if missing_cols:
            raise ValueError(f"Missing expected column(s): '{missing_cols}'.")

        # Overlap genetic and phenotype samples.
        self.overlap_samples(phenotypes[phenotypes_sample_id_column])

        # Reorder and select samples in the phenotypes dataset with respect to
        # their order in the genetic dataset.
        phenotypes = phenotypes.iloc[self.idx["phen"], :]

        # Standardize phenotypes if requested.
        if standardize_columns is not None:
            self.phenotype_scalers: Optional[List] = []
            for col in standardize_columns:
                scaler = StdVariableScaler(
                    col,
                    phenotypes[col].mean(skipna=True),
                    phenotypes[col].std(skipna=True)
                )
                phenotypes[col] = scaler.scale(phenotypes[col])

                self.phenotype_scalers.append(scaler)
        else:
            self.phenotype_scalers = None

        # Prepare tensors from the phenotypes df.
        self.exogenous_columns = exogenous_columns
        if exogenous_columns:
            self.exog: Optional[torch.Tensor] = torch.tensor(
                phenotypes.loc[:, exogenous_columns].values
            )
        else:
            self.exog = None

        self.endogenous_columns = endogenous_columns
        if endogenous_columns:
            self.endog: Optional[torch.Tensor] = torch.tensor(
                phenotypes.loc[:, endogenous_columns].values
            )
        else:
            self.endog = None

        # Standardize genotypes if requested.
        if standardize_genotypes:
            self.genotype_scaling_mean_std = self.compute_genotype_scaling()
        else:
            self.genotype_scaling_mean_std = None

    def compute_genotype_scaling(self):
        # Estimate the mean and standard deviation of the genotypes to allow
        # standardization on the fly.
        sample_size = min(len(self), _STANDARDIZE_MAX_SAMPLE)
        sample = np.random.choice(
            np.arange(len(self)), size=sample_size, replace=False
        )
        m = np.empty((sample_size, self.backend.get_n_variants()), dtype=float)

        for i, idx in enumerate(sample):
            m[i, :] = self.backend[idx].numpy()

        return (
            torch.tensor(np.nanmean(m, axis=0)),
            torch.tensor(np.nanstd(m, axis=0))
        )

    def standardized_genotypes_to_dosage(self, genotypes):
        if self.genotype_scaling_mean_std is None:
            raise RuntimeError("Genotypes were not standardized.")

        genotypes *= self.genotype_scaling_mean_std[1]
        genotypes += self.genotype_scaling_mean_std[0]
        return genotypes

    def __getitem__(self, idx):
        """Retrieve data at index.

        The return type is a batch which may or may not contain the following
        fields depending on the genotype scaling and user requested variables:

            - dosage
            - std_genotypes
            - exogenous
            - endogenous

        """
        # Get the genotypes from the backend.
        geno = self.backend[self.idx["geno"][idx]]
        geno_std = None

        # Apply the standardization if requested.
        if self.genotype_scaling_mean_std is not None:
            geno_std = geno - self.genotype_scaling_mean_std[0]
            geno_std /= self.genotype_scaling_mean_std[1]

            # Impute NA to mean (0).
            geno_std = torch.nan_to_num(geno_std, nan=0)

        out = [geno]

        if geno_std is None:
            _geno_batch_fac = BatchDosage
        else:
            _geno_batch_fac = BatchDosageStandardized
            out.append(geno_std)

        if self.exog is not None:
            has_exog = True
            out.append(self.exog[idx, :])
        else:
            has_exog = False

        if self.endog is not None:
            has_endog = True
            out.append(self.endog[idx, :])
        else:
            has_endog = False

        _batch_fac = BatchFactory.union(
            "PhenotypeDatasetBatch",
            _geno_batch_fac,
            _get_phenotype_batch_factory(has_exog, has_endog)
        )

        return _batch_fac(*out)

    def __len__(self):
        return len(self.idx["geno"])

    def overlap_samples(self, phenotype_samples):
        """Finds overlapping samples between genetic and phenotype dataset.

        Sets indexers:

            self.idx["geno"] is the indices in the genetic backend.
            self.idx["phen"] is the indices in the phenotype DF.

        """
        genetic_samples = self.backend.get_samples()
        overlap = set(genetic_samples) & set(phenotype_samples)

        genetic_samples_type = type(genetic_samples[0])
        phenotype_samples_type = type(phenotype_samples[0])
        if genetic_samples_type is not phenotype_samples_type:
            raise ValueError(
                f"Genetic file sample type: '{genetic_samples_type}' is "
                f"different from phenotype samples ('{phenotype_samples_type}'"
                ")."
            )

        if len(overlap) == 0:
            raise ValueError("No overlap between the genetic and phenotype "
                             "samples.")

        indexer = make_indexer(genetic_samples, phenotype_samples)
        self.idx = {
            "geno": indexer.left_idx.values,
            "phen": indexer.right_idx.values
        }


def make_indexer(left_samples, right_samples):
    left_df = pd.DataFrame({"left_id": left_samples})
    left_df["left_idx"] = np.arange(left_df.shape[0])

    right_df = pd.DataFrame({"right_id": right_samples})
    right_df["right_idx"] = np.arange(right_df.shape[0])

    df = pd.merge(left_df, right_df,
                  left_on="left_id", right_on="right_id",
                  how="inner")

    if df.shape[0] == 0:
        raise RuntimeError("Can't index non-overlapping datasets.")

    # Sort wrt left to minimize shuffling around on this dataset.
    df = df.sort_values("left_idx")

    return df
