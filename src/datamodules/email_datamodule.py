from typing import Optional
from torch.utils.data import DataLoader, Dataset
from src.datamodules.datasets.email_dataset import EmailDataset
from src.utils.email_process import load_raw_data


class EmailDataModule():

    def __init__(
        self,
        bert_name: str = "",
        batch_size: int = 32,
        num_workers: int = 0,
        data_dir: str = "",
        **kwargs
    ):
        super().__init__()
        self.bert_name = bert_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

    # @property
    # def num_classes(self) -> int:
    #     return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        train_raw, valid_raw, test_raw = load_raw_data(self.data_dir)
        #print(train_raw)
        #break
        self.data_train = EmailDataset(train_raw, self.bert_name)
        self.data_valid = EmailDataset(valid_raw, self.bert_name)
        self.data_test = EmailDataset(test_raw, self.bert_name)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.data_train.collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.data_valid.collate_func,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.data_test.collate_func,
        )
