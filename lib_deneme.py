# Make sure Libs are set correctly

import torch
print(torch.backends.mps.is_available())
print(torch.device("mps"))
