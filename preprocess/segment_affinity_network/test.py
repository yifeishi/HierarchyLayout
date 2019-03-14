import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset_affinity import *
from network_affinity import *
import util

config = util.get_args()
transformed_dataset = AffinityDataset(csv_file=config.test_csv_path, root_dir=config.train_root_dir)
dataloader = DataLoader(transformed_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

model = torch.load('./models/snapshots_2018-11-07_09-07-34/model54400.pkl')
model.cuda(config.gpu)

if not os.path.exists(config.feature_path):
    os.makedirs(config.feature_path)

count = 0
correct = 0
incorrect = 0
print('Start testing ......')
for i_batch, sample_batched in enumerate(dataloader):
    sample = transformed_dataset[i_batch]
    feature = sample_batched['feature'].float()
    label = sample_batched['label'].float()
    feature = Variable(feature.cuda(config.gpu), requires_grad=True)
    label = Variable(label.cuda(config.gpu), requires_grad=True)
    output = model(feature)
    outputSoftmax = nn.Softmax()(output).double()
    value, pred_label = torch.max(outputSoftmax, 1)
    lossAll = torch.cat([nn.CrossEntropyLoss()(l.unsqueeze(0), gt[None])[None].mul(0.2) for l, gt in zip(output, label.long())], 0)
    loss = torch.sum(lossAll)
    #print('pred_label',pred_label,'value',value,'label',label,'loss',loss)
    
    pred_label = pred_label.data.cpu().numpy()
    label = label.data.cpu().numpy()
    #print('pred_label',pred_label,'label',label)
    #print('')
    #print('')

    for i in range(len(pred_label)):
        if pred_label[i] == label[i]:
            correct+=1
        else:
            incorrect+=1
    count = count + 1
print('accuracy',float(correct)/(correct+incorrect))
print('Finished...')