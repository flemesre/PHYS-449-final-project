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
