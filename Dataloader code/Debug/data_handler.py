import numpy as np

if __name__ == '__main__':
    den_contrast = np.loadtxt('den_contrast_1.txt')
    print(den_contrast)
    print(den_contrast.shape)
    # np.savetxt('trial_den_contrast_1.txt', den_contrast)
