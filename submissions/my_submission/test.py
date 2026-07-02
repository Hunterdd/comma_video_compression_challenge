# pyrefly: ignore [missing-import]
import torch, matplotlib.pyplot as plt
state = torch.load('/Users/dhruv/Desktop/comma/comma_video_compression_challenge/submissions/my_submission/model/0.bin')
all_weights = torch.cat([v.flatten() for v in state.values() if v.dtype == torch.float32])
plt.hist(all_weights.numpy(), bins=200)
plt.savefig('weight_dist.png')