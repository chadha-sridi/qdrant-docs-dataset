from qdrant_sample_datasets.core.dataset import Dataset
from qdrant_sample_datasets.datasets.abo import AboDataset
from qdrant_sample_datasets.datasets.docs import DocsDataset
from qdrant_sample_datasets.datasets.esci import EsciDataset

DATASETS: dict[str, Dataset] = {
    dataset.name: dataset for dataset in (DocsDataset(), EsciDataset(), AboDataset())
}
