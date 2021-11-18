# Questions about the paper/implementation

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
"The initial weights of the kernels in a layer were set
following the Xavier initialization technique"

"The weights [of the fully-connected layers] were initialized using the same
Xavier initialization technique used for the kernel weights of
the convolutional layers."

But what about the biases?

## 4
For gamma in the Cauchy loss function, what should its initial value be? (Currently set as `1`.)

# Code

## 1
Is the rate of convergence ok?
```
iteration = 0   loss = 0.34011247754096985  test_loss = 0.32501065731048584  train time = 1.9062778949737549  test time = 0.015598535537719727
iteration = 10   loss = 0.18402111530303955  test_loss = 0.2044360488653183  train time = 3.5624704360961914  test time = 0.0
iteration = 20   loss = 0.3192734122276306  test_loss = 0.17551086843013763  train time = 3.4999547004699707  test time = 0.0
iteration = 30   loss = 0.2715625464916229  test_loss = 0.20931397378444672  train time = 3.453092098236084  test time = 0.0
iteration = 40   loss = 0.16409455239772797  test_loss = 0.27377867698669434  train time = 3.457240343093872  test time = 0.0
iteration = 50   loss = 0.20284710824489594  test_loss = 0.22480344772338867  train time = 3.4694488048553467  test time = 0.0
iteration = 60   loss = 0.3017294108867645  test_loss = 0.36841732263565063  train time = 3.453094482421875  test time = 0.0
iteration = 70   loss = 0.21176420152187347  test_loss = 0.14091169834136963  train time = 3.4687178134918213  test time = 0.0
iteration = 80   loss = 0.23910002410411835  test_loss = 0.3280853033065796  train time = 3.468723773956299  test time = 0.0
iteration = 90   loss = 0.24048678576946259  test_loss = 0.13808567821979523  train time = 3.46871280670166  test time = 0.0
iteration = 100   loss = 0.1610146164894104  test_loss = 0.2683800458908081  train time = 3.47703218460083  test time = 0.0
iteration = 110   loss = 0.1984553337097168  test_loss = 0.1551235020160675  train time = 3.515571117401123  test time = 0.015699148178100586
iteration = 120   loss = 0.26128289103507996  test_loss = 0.33866751194000244  train time = 3.5030062198638916  test time = 0.0
iteration = 130   loss = 0.3674178719520569  test_loss = 0.1735125333070755  train time = 3.5102651119232178  test time = 0.0
iteration = 140   loss = 0.19272185862064362  test_loss = 0.30309736728668213  train time = 3.5155982971191406  test time = 0.0
iteration = 150   loss = 0.0995434895157814  test_loss = 0.18976089358329773  train time = 3.4843382835388184  test time = 0.0
iteration = 160   loss = 0.13454009592533112  test_loss = 0.2440444529056549  train time = 3.515578031539917  test time = 0.0
iteration = 170   loss = 0.24846403300762177  test_loss = 0.1914767026901245  train time = 3.515751838684082  test time = 0.0
iteration = 180   loss = 0.10354619473218918  test_loss = 0.21512943506240845  train time = 3.546858310699463  test time = 0.0
iteration = 190   loss = 0.2294943779706955  test_loss = 0.28322121500968933  train time = 3.5090830326080322  test time = 0.0
iteration = 200   loss = 0.10325255244970322  test_loss = 0.1312413513660431  train time = 3.515611171722412  test time = 0.0
iteration = 210   loss = 0.1961805522441864  test_loss = 0.34607648849487305  train time = 3.515613317489624  test time = 0.0
iteration = 220   loss = 0.16982382535934448  test_loss = 0.18669569492340088  train time = 3.5391123294830322  test time = 0.0
iteration = 230   loss = 0.139402374625206  test_loss = 0.29125726222991943  train time = 3.499955177307129  test time = 0.0
iteration = 240   loss = 0.26615315675735474  test_loss = 0.3090594410896301  train time = 3.5468332767486572  test time = 0.0
iteration = 250   loss = 0.18852613866329193  test_loss = 0.1668270230293274  train time = 3.533890724182129  test time = 0.0
iteration = 260   loss = 0.23883633315563202  test_loss = 0.5040185451507568  train time = 3.5155835151672363  test time = 0.0
iteration = 270   loss = 0.15511399507522583  test_loss = 0.15989062190055847  train time = 3.515587091445923  test time = 0.0
iteration = 280   loss = 0.2016056776046753  test_loss = 0.32143184542655945  train time = 3.5156266689300537  test time = 0.0
iteration = 290   loss = 0.1478722095489502  test_loss = 0.21666219830513  train time = 3.5157768726348877  test time = 0.0
iteration = 300   loss = 0.2844078242778778  test_loss = 0.18856799602508545  train time = 3.528524875640869  test time = 0.0
iteration = 310   loss = 0.1640767604112625  test_loss = 0.256283700466156  train time = 3.5182275772094727  test time = 0.0
iteration = 320   loss = 0.24649620056152344  test_loss = 0.23522964119911194  train time = 3.562471866607666  test time = 0.0
iteration = 330   loss = 0.1347023993730545  test_loss = 0.39374101161956787  train time = 3.5312066078186035  test time = 0.0
iteration = 340   loss = 0.26819977164268494  test_loss = 0.13099034130573273  train time = 3.531202793121338  test time = 0.0
iteration = 350   loss = 0.1481616497039795  test_loss = 0.30274754762649536  train time = 3.5155720710754395  test time = 0.0
iteration = 360   loss = 0.24899636209011078  test_loss = 0.2645050585269928  train time = 3.546832323074341  test time = 0.0
iteration = 370   loss = 0.14734043180942535  test_loss = 0.19513195753097534  train time = 3.533019781112671  test time = 0.0
iteration = 380   loss = 0.1843460649251938  test_loss = 0.2153889536857605  train time = 3.5781099796295166  test time = 0.0
```

# Details
## 1
Output of the final conv layer/input to the first fully connected layer

Instead of calculating this from first principles, by printing out the shape of the output tensor

It looks like `fc input shape = torch.Size([64, 128, 2, 2, 2])`?

so the input for the 1st FC layer = 128 *2 *2 *2 = 1024

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
