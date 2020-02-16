# Accuracy(max) = 99.54(19th epoch) 
 
 
 ## model
 ```python
 class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias = False) # 28 > 26 | 3
        self.b1 = nn.BatchNorm2d(8)
        self.d1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias = False)  # 26 > 24 | 5
        self.b2 = nn.BatchNorm2d(8)
        self.d2 = nn.Dropout2d(0.1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1, bias = False) # 24 > 22 | 7
        self.b3 = nn.BatchNorm2d(16)
        self.d3 = nn.Dropout2d(0.1)
        self.pool1 = nn.MaxPool2d(2, 2)                           # 22 > 11


        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, bias = False) # 11 > 9 | 9
        self.b4 = nn.BatchNorm2d(16)
        self.d4 = nn.Dropout2d(0.1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1, bias = False)  # 9 > 7 | 11
        self.b5 = nn.BatchNorm2d(16)
        self.d5 = nn.Dropout2d(0.1)
        self.conv6 = nn.Conv2d(16, 24, 3, padding=1, bias = False)  # 7 > 5 | 13
        self.b6 = nn.BatchNorm2d(24)
        self.d6 = nn.Dropout2d(0.1)
        self.conv7 = nn.Conv2d(24, 24, 3, padding=1, bias = False) # 5 > 3  | 15
        self.b7 = nn.BatchNorm2d(24)
        self.d7 = nn.Dropout2d(0.1)
        self.conv1c = nn.Conv2d(24, 10, 3, bias = False)           # 3 > 1 | 17
        self.gap = nn.AdaptiveAvgPool2d((1,1))                     # 1 > 1 | 17

    def forward(self, x):
        x = self.pool1(self.d3(self.b3(F.relu(self.conv3(self.d2(self.b2(F.relu(self.conv2(self.d1(self.b1(F.relu(self.conv1(x)))))))))))))
        x = self.d5(self.b5(F.relu(self.conv5(self.d4(self.b4(F.relu(self.conv4(x))))))))
        x = self.d6(self.b6(F.relu(self.conv6(x))))
        x = F.avg_pool2d(x,(4,4))
        x = self.conv1c(self.d7(self.b7(F.relu(self.conv7(x)))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x)
 
```

## model params


        Layer (type)               Output Shape         Param #

            Conv2d-1            [-1, 8, 28, 28]              72
       BatchNorm2d-2            [-1, 8, 28, 28]              16
         Dropout2d-3            [-1, 8, 28, 28]               0
            Conv2d-4            [-1, 8, 28, 28]             576
       BatchNorm2d-5            [-1, 8, 28, 28]              16
         Dropout2d-6            [-1, 8, 28, 28]               0
            Conv2d-7           [-1, 16, 28, 28]           1,152
       BatchNorm2d-8           [-1, 16, 28, 28]              32
         Dropout2d-9           [-1, 16, 28, 28]               0
        MaxPool2d-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           2,304
      BatchNorm2d-12           [-1, 16, 14, 14]              32
        Dropout2d-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,304
      BatchNorm2d-15           [-1, 16, 14, 14]              32
        Dropout2d-16           [-1, 16, 14, 14]               0
           Conv2d-17           [-1, 24, 14, 14]           3,456
      BatchNorm2d-18           [-1, 24, 14, 14]              48
        Dropout2d-19           [-1, 24, 14, 14]               0
           Conv2d-20             [-1, 24, 3, 3]           5,184
      BatchNorm2d-21             [-1, 24, 3, 3]              48
        Dropout2d-22             [-1, 24, 3, 3]               0
           Conv2d-23             [-1, 10, 1, 1]           2,160
AdaptiveAvgPool2d-24             [-1, 10, 1, 1]               0



Total params: 17,432
Trainable params: 17,432
Non-trainable params: 0

Input size (MB): 0.00
Forward/backward pass size (MB): 0.85
Params size (MB): 0.07
Estimated Total Size (MB): 0.92



 ## Epoch Logs
 
 
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:37: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.06407123804092407 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.35it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0410, Accuracy: 9872/10000 (99%)

loss=0.03049006499350071 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.95it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0302, Accuracy: 9906/10000 (99%)

loss=0.1060747504234314 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.08it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0260, Accuracy: 9915/10000 (99%)

loss=0.010345171205699444 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.07it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0194, Accuracy: 9939/10000 (99%)

loss=0.042034074664115906 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.96it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0201, Accuracy: 9938/10000 (99%)

loss=0.03167784586548805 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.14it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0187, Accuracy: 9939/10000 (99%)

loss=0.005409757141023874 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.19it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9935/10000 (99%)

loss=0.04193221032619476 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.88it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0179, Accuracy: 9936/10000 (99%)

loss=0.04351457953453064 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.25it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0176, Accuracy: 9942/10000 (99%)

loss=0.028721177950501442 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.13it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0170, Accuracy: 9939/10000 (99%)

loss=0.049138277769088745 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.23it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0178, Accuracy: 9948/10000 (99%)

loss=0.0081541882827878 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0158, Accuracy: 9945/10000 (99%)

loss=0.029937485232949257 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.16it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0166, Accuracy: 9942/10000 (99%)

loss=0.00937699805945158 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.95it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0179, Accuracy: 9939/10000 (99%)

loss=0.005378881935030222 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.78it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0157, Accuracy: 9944/10000 (99%)

loss=0.0012924770126119256 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.09it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0177, Accuracy: 9937/10000 (99%)

loss=0.016202306374907494 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.78it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0151, Accuracy: 9950/10000 (100%)

loss=0.03573565557599068 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.49it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0157, Accuracy: 9942/10000 (99%)

loss=0.0022128920536488295 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.00it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0135, Accuracy: 9954/10000 (100%)

loss=0.006376375909894705 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.93it/s]

Test set: Average loss: 0.0147, Accuracy: 9950/10000 (100%)
