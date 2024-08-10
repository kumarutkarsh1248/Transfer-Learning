import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Linear(4096, 3),
    nn.Softmax(inplace=True),
)




# utility methods
def resized_image(image, target_size=(224, 224)):
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Resize the image
    image_resized = cv.resize(image, (target_width, target_height))
    # image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)


    # Calculate the scale factors
    x_scale = target_width / original_width
    y_scale = target_height / original_height

    return image_resized

train_data_features_A = []
train_data_features_B = []
train_data_features_C = []

data_features = []
data_labels = []

for i in range(1, 60):
    image_path =  f'personA/frame{i*6}.jpg'
    image = cv.imread(image_path)
    image = resized_image(image, target_size=(224, 224))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    data_features.append(image)
    data_labels.append(0)


    image_path =  f'personB/frame{i*6}.jpg'
    image = cv.imread(image_path)
    image = resized_image(image, target_size=(224, 224))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    data_features.append(image)
    data_labels.append(1)


    image_path =  f'personC/frame{i*6}.jpg'
    image = cv.imread(image_path)
    image = resized_image(image, target_size=(224, 224))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    data_features.append(image)
    data_labels.append(2)

data_features = np.array(data_features)
data_labels = np.array(data_labels)


data_features.shape
train_data_features = data_features[0:160]
train_data_labels = data_labels[0:160]

test_data_features = data_features[160:]
test_data_labels = data_labels[160:]



# creating the data loader
# /content/drive/MyDrive/my projects/face detection/image
class CustData(torch.utils.data.Dataset):
    def __init__(self, train_X, train_y):     #constructor function to initialize the variable
        self.train_X = train_X
        self.train_y = train_y

    def __len__(self):  # this funtion should be defined in such a way that it should return number of sample points
        return self.train_X.shape[0]


    def __getitem__(self, idx): # it should be defined such that it return the one sample point so that pytorch can process
        dp = self.train_X[idx]  #data point
        train_y = self.train_y[idx]
        dp = torch.tensor(dp, dtype=torch.float32)
        # dp = dp.unsqueeze(dim=0) # this image has only one channel
        train_y = torch.tensor(train_y, dtype=torch.long)
        return dp, train_y



# here we are creating the dataset
cd_train = CustData(train_data_features, train_data_labels)
cd_test = CustData(test_data_features, test_data_labels)



batch_size = 10
dl_train = torch.utils.data.DataLoader(cd_train,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 num_workers = 1,   #for parallel processing
                                 pin_memory = False, # when we have gpu
                                #  sampler = None,
                                #  collate_fn = new_collate,
                                )
dl_test = torch.utils.data.DataLoader(cd_test,
                                 batch_size = 17,
                                 shuffle = True,
                                 num_workers = 1,   #for parallel processing
                                 pin_memory = False, # when we have gpu
                                #  sampler = None,
                                #  collate_fn = new_collate,
                                )


import torch.optim as optim

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.1)
loss_fn = nn.CrossEntropyLoss()



epochs = 2
loss_value = []
total_batches = (epochs * len(train_data_features) // batch_size)

for epoch in range(epochs):
    # set the model to training mode

    for batch_idx, (data, target) in enumerate(dl_train):
        data, target = data.to(device), target.to(device)
        model.train() # turn on gradient tracking, it has computational signifance



        # 1. forward pass
        y_pred = model(data)

        # 2. calculate the loss
        loss = loss_fn(y_pred, target)

        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. perform backpropagation in the loss with respect to the parameters of the model_1
        loss.backward()

        # 5. step the optimizer (perform gradient descent )
        optimizer.step()

        model.eval() # turns off different setting in the model_1 not needed for evaluation/testing

        with torch.inference_mode(): # turn off gradient tracking and couple of more thing behind the scene

            test_loss = 0
            correct = 0
            for data, target in dl_train:
                data, target = data.to(device), target.to(device)

                # 1. do the forward pass
                test_pred = model(data)

                # 2. calculate the loss
                test_loss += loss_fn(test_pred, target).item()

                # 3.accuracy
                pred = test_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            # printing out whats happening
            # if(batch_idx % 10 == 0):
            test_loss = test_loss/len(dl_test)
            print(f'epoch: {epoch}, trained on: {(batch_idx+1) * batch_size}/{len(train_data_features)}, test_loss = {test_loss:.15f}, correct: {correct}/{len(test_data_features)}')














