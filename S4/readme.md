# Train Accuracy(max) = 99.54(19th epoch) 
 
 
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

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
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
================================================================
Total params: 17,432
Trainable params: 17,432
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.85
Params size (MB): 0.07
Estimated Total Size (MB): 0.92
----------------------------------------------------------------


 ## Epoch Logs
 
 
 0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:38: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.10238625854253769 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.83it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0467, Accuracy: 9858/10000 (99%)

loss=0.04741235077381134 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.28it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0320, Accuracy: 9894/10000 (99%)

loss=0.12687800824642181 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.32it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0322, Accuracy: 9892/10000 (99%)

loss=0.04758177697658539 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 39.55it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0273, Accuracy: 9911/10000 (99%)

loss=0.029237249866127968 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 39.76it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0230, Accuracy: 9924/10000 (99%)

loss=0.061964284628629684 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 40.81it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0256, Accuracy: 9915/10000 (99%)

loss=0.03135301545262337 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.57it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0227, Accuracy: 9929/10000 (99%)

loss=0.1020083948969841 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 36.54it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0225, Accuracy: 9930/10000 (99%)

loss=0.03699042275547981 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 43.74it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0216, Accuracy: 9925/10000 (99%)

loss=0.01206710934638977 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.24it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0181, Accuracy: 9931/10000 (99%)

loss=0.06720546633005142 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.10it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0198, Accuracy: 9926/10000 (99%)

loss=0.0034883618354797363 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 39.70it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0181, Accuracy: 9938/10000 (99%)

loss=0.029300441965460777 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 39.80it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0182, Accuracy: 9932/10000 (99%)

loss=0.0033511221408843994 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 40.02it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9934/10000 (99%)

loss=0.013589486479759216 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.27it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9936/10000 (99%)

loss=0.08053451776504517 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 40.38it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0175, Accuracy: 9943/10000 (99%)

loss=0.01174600888043642 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 39.87it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9941/10000 (99%)

loss=0.027496641501784325 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.04it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0188, Accuracy: 9943/10000 (99%)

loss=0.030316561460494995 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 39.55it/s]

Test set: Average loss: 0.0178, Accuracy: 9940/10000 (99%)
