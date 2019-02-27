from sparse_dataset import SparseDataset
from torch.utils.data import DataLoader


class SparseDataLoader(DataLoader):
    def __init__(self, flags):
        self._flags = flags
        self.dataset = SparseDataset(flags)
        super(SparseDataLoader, self).__init__(self.dataset,
                          batch_size = flags.MINIBATCH_SIZE * len(flags.GPUS),
                          shuffle=True,
                          num_workers=flags.NUM_WORKERS,
                          collate_fn=self.dataset.collate_fn,
                          drop_last=True)

    def num_entries(self):
        return len(self.dataset)

    def num_channels(self):
        return self.dataset.num_channels()

    def cycle(self):
        while True:
            for x in self:
                yield x
