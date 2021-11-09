# Data

data looks like

3.063454018339792188e+13

8.097707056114916992e+11

this looks like the dark matter halo mass M/M_stellar

(Recall log(M/M_stellar) in [11,13.4])(log 10?)

each number repeats several times, so if this is the halo mass, it makes sense, since each halo has many particles, so they should all share the same halo mass

## handling the file

opening .npy files

https://stackoverflow.com/questions/53084637/how-do-you-open-npy-files?fbclid=IwAR0JfRhrgDDfHVfu4WFdl434iwSmeXFZ6dBadcA0xqBmAvvp5vSe-SMxKAo

the file is loaded as a numpy array?

Question: how large can an array get?

## Input to the CNN: 

this format? (N, C_{in}, D, H, W)

https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
