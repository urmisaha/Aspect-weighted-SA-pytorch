# Errors encountered while running model (Possible solutions are noted down here)

1. train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
File ".. .. /lib/python3.6/site-packages/torch/utils/data/dataset.py", line 36, in __init__
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
AssertionError

> **Solution:** size of train and test dataset may be unequal in preprocessing file. Fix and run preprocessing file again before running model
