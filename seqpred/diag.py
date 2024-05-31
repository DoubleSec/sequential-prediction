from lightning.pytorch.callbacks import Callback
import torch


class MemoryMonitorCallback(Callback):

    def __init__(self, snapshot_path):
        super().__init__()
        self.snapshot_path = snapshot_path

    def on_train_start(self, trainer, module):
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    def on_train_epoch_end(self, trainer, module):
        torch.cuda.memory._dump_snapshot(self.snapshot_path)

    def on_train_end(self, trainer, module):
        torch.cuda.memory._record_memory_history(enabled=None)
