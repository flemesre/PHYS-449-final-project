## 1

Is it bad to save the ndarrays as tensors after preprocessing?

Tensors are only float32? Less precision?

Can't use np.savetxt for higher dimensional arrays

Nevermind, torch does support float64 tensors on cpus

## 2

Since I haven't figured out how to import torch when using WSL

I have to use a separate program to do the torch processing stuff

That means I need an intermediate program to convert `.gadget3` stuff into sth np can read

## 3

If we want to create an input tensor with all the subboxes processed

ie a tensor with size `(num_particles,75,75,75)`, it will need `51.5 TiB`; not possible

So we will have to compute the subboxes every time it is referenced during training?
