# Questions about the paper/implementation

## 1
the paper says "The feature maps are then fed
to max-pooling layers, which reduce their dimensionality by
taking the average over 2 × 2 × 2 non-overlapping regions of
the feature maps."

so average or max?

Finn: I think that's an error. Since they said everywhere else that they used max pooling, it must be max pooling.

## 2
Does the last conv layer has leaky relu?

"After the sixth convolutional layer and subsequent pooling
layer, the output is flattened into a one-dimensional vector and
fed to a series of 3 fully-connected layers"

"All convolutional layers (but the first one) are followed by
max-pooling layers; their output is then used as input to the
non-linear leaky rectified linear unit (LeakyReLU) [51] activation function"

Finn: Maybe we can try with and without LeakyReLU on the last layer?

## 3
"The initial weights of the kernels in a layer were set
following the Xavier initialization technique"

"The weights [of the fully-connected layers] were initialized using the same
Xavier initialization technique used for the kernel weights of
the convolutional layers."

But what about the biases?

Finn: Pretty sure it is standard to initialize biases as 0, so if nothing is said about it in the paper, that is likely what was done.

## 4
For gamma in the Cauchy loss function, what should its initial value be? (Currently set as `1`.)

# Code

## 1

# Details
## 1
Output of the final conv layer/input to the first fully connected layer

Instead of calculating this from first principles, by printing out the shape of the output tensor

It looks like `fc input shape = torch.Size([64, 128, 2, 2, 2])`?

so the input for the 1st FC layer = 128 *2 *2 *2 = 1024

## 2
To match the shape of the output tensor with that of the input tensor

Need to go from (batch size,) to (batch size, 1)

Hence `torch.unsqueeze(_true_mass, 1)`

# Trivia
## 1
`torch.cuda.empty_cache()` seems to clear the memory a little? (judging from task manager)

# Resolved
## 1
can't do both data processing and dataloading on my gpu?

```
RuntimeError: CUDA out of memory. Tried to allocate 3.22 GiB (GPU 0; 8.00 GiB total capacity; 3.94 GiB already allocated; 2.09 GiB free;  3.95 GiB reserved in total by PyTorch)
```

in fact just the density field + neural net is too much?
```
RuntimeError: CUDA out of memory. Tried to allocate 3.22 GiB (GPU 0; 8.00 GiB total capacity; 3.41 GiB already allocated; 2.61 GiB free; 3.43 GiB reserved in total by PyTorch)
```
Solution: set smaller batch size (8 or 10) and number of test particles (8)
