import numpy as np
import torch


device = torch.device("cpu")

class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.input_data = torch.tensor(np.zeros((100,3,3))).to(device)
        self.output_data = torch.tensor(np.zeros(100)).to(device)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return self.input_data[idx, :, :], self.output_data[idx]

train_dataset = TrainingDataset()
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# for stuff in enumerate(train_dataloader): #batch, (_x, _y)
#     print(f"{stuff}")

for batch, (_x, _y) in enumerate(train_dataloader):
    print(f"batch = {batch}   _x shape = {_x.shape}   _y shape = {_y.shape}")


# a = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# torch.save(a, 'a.pt')

# a = torch.load('a.pt')
# print(a)

# np.savetxt('trial_den_contrast_1.txt', a)