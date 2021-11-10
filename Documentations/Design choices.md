## 1

Is it bad to save the ndarrays as tensors after preprocessing?

Tensors are only float32? Less precision?

Can't use np.savetxt for higher dimensional arrays

Nevermind, torch does support float64 tensors on cpus

## 2

Since I haven't figured out how to import torch when using WSL

I have to use a separate program to do the torch processing stuff

That means I need an intermediate program to convert `.gadget3` stuff into sth np can read
