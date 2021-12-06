import torch
import torch.nn as nn
import torch.nn.functional as func
class Net(nn.Module):
    # based on architecture in :
    #https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

    def __init__(self):
        super(Net,self).__init__()
        '''
            Convolution Neural Network with the following structure

            INPUT: 14x14x1 Density fild
            LRELU+CONV1; Convolves image into 10 channels, kernel size 3
            LRELU+CONV2; Convolves image into 15 channels, kernel size 3
            POOL1; MAX POOLING LAYER, kernel size 2

            FLATTENS ARRAY, FEEDS INTO FC NETWORK

            relu+FC1: INPUT OF 5*5*15 UNITS, OUTPUT OF 100
            relu+FC2: INPUT OF 100 UNITS, OUTPUT OF 80
            relu+FC3: INPUT OF 80 UNITS, OUTPUT OF 25
            FC4: INPUT OF 25 UNITS, OUTPUT OF 5
            Run through a softmax layer before being passed

        '''
        # 864 x 864 x 864 x n_cx, filter of 3
                # 864 - 3+ 1 ---> 862
        self.conv1= nn.Conv3d(1,32,3) # (864x864x864x1) -->(862x862x862x32) # 14-3/1 +1 (12)
        self.conv2 = nn.Conv3d(32,32,3) # (862x862x862x32)-->(860x860x860x32) 861 - 3+1
        self.pool1 = nn.MaxPool3d(3) # 858x858x857x32

        self.conv3 = nn.Conv3d(32,64,3) # 857x857x857x32 -->855x855x855x64
        self.pool2 = nn.MaxPool3d(3) # 853x853x853x64

        self.conv4 = nn.Conv3d(64,128,3) # (853x853x853x64)-->(851x851x851x128)
        self.pool3 = nn.MaxPool3d(2) # 849x849x849x128

        self.conv5 = nn.Conv3d(128,128,3) # (849x849x849x128 -->(847x847x847x128)
        self.pool4 = nn.MaxPool3d(3) # 845x845x845x128

        self.conv6 = nn.Conv3d(128,128,3) # 845x845x845x128 -->843x843x843x128
        self.pool5 = nn.MaxPool3d(2) # 842x842x842x128



        self.fc1 = nn.Linear(842*842*842*128,256) # some way to get self.size?
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        #self.softmax = nn.Softmax(dim=0)


    def forward(self,x):
        #print(x.shape)

        c1 = func.relu(self.conv1(x.float()))

        c2 = func.relu(self.conv2(c1))
        p1 = self.pool1(c2)

        c3 = func.relu(self.conv3(p1))
        p2 = self.pool2(c3)

        c4 = func.relu(self.conv4(p2))
        p3 = self.pool(c4)

        c5 = func.relu(self.conv5(p3))
        p4 = self.pool(c5)

        c6 =func.relu(self.conv6(p4))
        p5 =self.pool(c6)



        flats= p5.view(-1,5*5*15)

        f1 = func.relu(self.fc1(flats))

        f2=func.relu(self.fc2(f1))
        f3= self.fc3(f2)

        #f4 = self.fc4(f3)#func.relu(self.fc3(fullc2))
        #f4  = self.softmax(f4)

        return f4


    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.conv6.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


    def backprop(self, inputs,targets, loss, epoch, optimizer):
        self.train()

        prediction = self.forward(inputs)
        obj_val= loss(prediction, targets)

        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item(),prediction

    def test(self, inputs,targets, loss, epoch):
        '''
        INPUTS: inuts ---> data to test with
                targets ---> known truth values to evaluate with
                loss ---> chosen loss function
                epoch ---> epoch/iteration of training

        OUTPUTS: latest computed loss from test data
                  prediction --> array of predicted digits



        '''



        self.eval()
        with torch.no_grad():

            prediction = self.forward(inputs)
            cross_val= loss(prediction, targets)
        return cross_val.item(),prediction

# ask about implementing CNN with images. Why can't i iterate over them

#estMNIST = Net().to(torch.device("cpu"))
class NetSkip(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        '''
            Convolution Neural Network with the following structure
            + SKIP CONNECTIONs?????
            INPUT: 14x14x1 Density fild
            LRELU+CONV1; Convolves image into 10 channels, kernel size 3
            LRELU+CONV2; Convolves image into 15 channels, kernel size 3
            POOL1; MAX POOLING LAYER, kernel size 2

            FLATTENS ARRAY, FEEDS INTO FC NETWORK

            relu+FC1: INPUT OF 5*5*15 UNITS, OUTPUT OF 100
            relu+FC2: INPUT OF 100 UNITS, OUTPUT OF 80
            relu+FC3: INPUT OF 80 UNITS, OUTPUT OF 25
            FC4: INPUT OF 25 UNITS, OUTPUT OF 5
            Run through a softmax layer before being passed

        '''
        self.conv1= nn.Conv3d(1,10,3) # (14x14x1) -->(10x10x10) # 14-3/1 +1 (12)
        self.conv2 = nn.Conv3d(10,15,3) # (10x10x10)-->(6x6x15)
        self.pool1 = nn.MaxPool3d(2) # 3x3x15

        self.conv3 = nn.Conv3d(10,15,3) # (10x10x10)-->(6x6x15)
        self.pool2 = nn.MaxPool3d(2) # 3x3x15

        self.conv4 = nn.Conv3d(10,15,3) # (10x10x10)-->(6x6x15)
        self.pool3 = nn.MaxPool3d(2) # 3x3x15

        self.conv5 = nn.Conv3d(10,15,3) # (10x10x10)-->(6x6x15)
        self.pool4 = nn.MaxPool3d(2) # 3x3x15

        self.conv6 = nn.Conv3d(10,15,3) # (10x10x10)-->(6x6x15)
        self.pool5 = nn.MaxPool3d(2) # 3x3x15



        self.fc1 = nn.Linear(5*5*15,100)
        self.fc2 = nn.Linear(100,80)
        self.fc3 = nn.Linear(80,25)

        #self.softmax = nn.Softmax(dim=0)


    def forward(self,x):
        #print(x.shape)

        c1 = func.relu(self.conv1(x.float()))

        c2 = func.relu(self.conv2(c1))
        p1 = self.pool1(c2)

        c3 = func.relu(self.conv3(p1))
        p2 = self.pool2(c3)

        c4 = func.relu(self.conv4(p2))
        p3 = self.pool(c4)

        c5 = func.relu(self.conv5(p3))
        p4 = self.pool(c5)

        c6 =func.relu(self.conv6(p4))
        p5 =self.pool(c6)



        flats= p5.view(-1,5*5*15)

        f1 = func.relu(self.fc1(flats))

        f2=func.relu(self.fc2(f1))
        f3= self.fc3(f2)

        #f4 = self.fc4(f3)#func.relu(self.fc3(fullc2))
        #f4  = self.softmax(f4)

        return f4


    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.conv6.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


    def backprop(self, inputs,targets, loss, epoch, optimizer):
        self.train()

        prediction = self.forward(inputs)
        obj_val= loss(prediction, targets)

        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item(),prediction

    def test(self, inputs,targets, loss, epoch):
        '''
        INPUTS: inuts ---> data to test with
                targets ---> known truth values to evaluate with
                loss ---> chosen loss function
                epoch ---> epoch/iteration of training

        OUTPUTS: latest computed loss from test data
                  prediction --> array of predicted digits



        '''



        self.eval()
        with torch.no_grad():

            prediction = self.forward(inputs)
            cross_val= loss(prediction, targets)
        return cross_val.item(),prediction
