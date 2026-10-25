from qdrant_sample_datasets.core.dataset import Dataset
from qdrant_sample_datasets.datasets.docs import DocsDataset

DATASETS: dict[str, Dataset] = {dataset.name: dataset for dataset in (DocsDataset(),)}
