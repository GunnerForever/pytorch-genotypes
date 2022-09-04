# Datasets

We provide datasets that are compatible with [``torch.utils.data.Dataset``](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Dataset implementations are typically done by wrapping a ``GeneticDatasetBackend`` which provides an abstraction, and possible performance optimisation, over the underlying source of genotypes.

## Genetic datasets

Genetic datasets are useful to train models of genotype data. For example, they may be useful for imputation tasks, or for dimensionality reduction capturing global ancestry (_e.g._ principal component analysis).

One important functionality that is offered by genetic datasets is the ability to do variant standardization. For example, some losses may require genotype probabilities whereas some models may perform better when the input variants have been standardized. Genetic datasets provide an easy way of having access to both standardized and additively-encoded variants during training.

## Phenotype datasets

For tasks where the genotype-phenotype relationship is of interest, ``PhenotypeGeneticDataset`` may be used. This class takes care of joining the genetic and phenotypic datasets.
