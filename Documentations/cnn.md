## 1
the paper says "The feature maps are then fed
to max-pooling layers, which reduce their dimensionality by
taking the average over 2 × 2 × 2 non-overlapping regions of
the feature maps."

so average or max?

## 2
Does the last conv layer has leaky relu?

"After the sixth convolutional layer and subsequent pooling
layer, the output is flattened into a one-dimensional vector and
fed to a series of 3 fully-connected layers"

"All convolutional layers (but the first one) are followed by
max-pooling layers; their output is then used as input to the
non-linear leaky rectified linear unit (LeakyReLU) [51] activation function"

## 3
can't do both data processing and dataloading on my gpu?

```
RuntimeError: CUDA out of memory. Tried to allocate 3.22 GiB (GPU 0; 8.00 GiB total capacity; 3.94 GiB already allocated; 2.09 GiB free;  3.95 GiB reserved in total by PyTorch)
```

in fact just the density field + neural net is too much?

## 4
it looks like `fc input shape = torch.Size([64, 128, 2, 2, 2])`?