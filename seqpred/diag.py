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


def rollout(attentions, head_fusion):
    # Attentions are n x h x o x i, maybe?
    result = torch.eye(attentions[0].size(-1))

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.amax(dim=1)
            elif head_fusion == "min":
                attention_heads_fused = attention.amin(dim=1)
            else:
                raise "Attention head fusion type Not supported"

            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused)
            # What is the point of multiplying by 1?
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            # result is n x o x i
            result = torch.matmul(a, result.to(a))

    # Look at the total attention between the class token,
    # and the image patches
    return result.squeeze(dim=0)[-1, :]
