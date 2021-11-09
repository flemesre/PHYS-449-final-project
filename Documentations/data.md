# Data

Source: https://console.cloud.google.com/storage/browser/deep-halos-data/full-data?cloudshell=false&hl=en-AU&project=deephalos&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false

## Questions
Which file = IC?

How to read .gadget3 files?

In dlhalos_code/data_processing.py, used: 
```
pynbody.load(path1 + "IC_doub_z99_256.gadget3")
```

How to install pynbody?

What is the training_simulation folder about?

## Interpretation of data

data looks like

3.063454018339792188e+13

8.097707056114916992e+11

this looks like the dark matter halo mass M/M_stellar

(Recall log(M/M_stellar) in [11,13.4])(log 10?)

each number repeats several times, so if this is the halo mass, it makes sense, since each halo has many particles, so they should all share the same halo mass

## Data format

eg data_reseed1_simulation_reseed1_halo_mass_particles.npy

the shape of the array is (16777216,)    (ie 1D array)

Note that 256^3 = 16777216, this is consistent with the fact that each simulation has N = 256^3 dark matter particles.

## Handling the file

opening .npy files

https://stackoverflow.com/questions/53084637/how-do-you-open-npy-files?fbclid=IwAR0JfRhrgDDfHVfu4WFdl434iwSmeXFZ6dBadcA0xqBmAvvp5vSe-SMxKAo

the file is loaded as a numpy array?

Question: how large can an array get?

## Input to the CNN: 

this format? (N, C_{in}, D, H, W)

https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

## Logistics

pip install on windows doesn't work

What u need to do:

get WSL

https://docs.microsoft.com/en-us/windows/wsl/install?fbclid=IwAR1VRQ9E1ui9wsLqiPP51mFS-T0h4JWYRCi6HYGRx6tjA6i-bHoGOFRa7so

ubuntu will run automatically upon restart, set up account and password

then use pip install pynbody

to run code through PyCharm (installed on Windows), use

https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#wsl-terminal

Note that wsl.exe is at 
```
C:\Windows\System32\wsl.exe
```

## Cut and paste

(This is only on my machine)

```
Python 3 /mnt/d/Downloads/n8t.py
```
