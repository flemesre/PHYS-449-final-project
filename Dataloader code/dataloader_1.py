import numpy as np
import torch, torchvision

if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = ''

    device = torch.device("cpu")

    # do I need to use open() for better file handling?

    try:
        print("Attempting to load norm_den_contrast_1.pt")
        norm_den_contrast = torch.load(path+'norm_den_contrast_1.pt')
        print("Loaded normalized density contrast")
        print(norm_den_contrast)

    except OSError: #FileNotFoundError
        print('norm_den_contrast_1.pt not found')
        print('Loading den_contrast_1.txt')
        den_contrast = torch.tensor(np.loadtxt(path+'den_contrast_1.txt')).to(device)
        norm_den_contrast = (den_contrast - torch.mean(den_contrast))/torch.std(den_contrast)
        print(norm_den_contrast)

        torch.save(norm_den_contrast, path+'norm_den_contrast_1.pt')
        print('norm_den_contrast_1.py created')
