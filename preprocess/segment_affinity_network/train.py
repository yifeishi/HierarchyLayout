import os
import torch
import torchvision
import torch.nn as nn

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from time import gmtime, strftime, clock
from dataset_affinity import *
from network_affinity import *
import util

config = util.get_args()
transformed_dataset = AffinityDataset(csv_file=config.train_csv_path, root_dir=config.train_root_dir)
dataloader = DataLoader(transformed_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

model = SiameseNetwork()
model.cuda(config.gpu)

triplet_loss = nn.TripletMarginLoss(margin=1.0)
learning_rate = config.lr
iteration = 0

if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

print('Start training ......')
for epoch in range(config.epochs):
    optimizer = optim.Adam(model.parameters())
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
        #print('pred_label',pred_label,'value',value,'label',label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d, %d]   lr: %.6f   loss: %.10f' %(epoch + 1, i_batch + 1, learning_rate, loss))

        if config.save_snapshot and iteration % config.save_snapshot_every == 0 :
            print('Saving snapshots of the models ...... ')
            torch.save(model, snapshot_folder+'/model'+ str(iteration) + '.pkl')
        if iteration % config.lr_decay_every == config.lr_decay_every - 1:
            learning_rate = learning_rate * config.lr_decay_by
        iteration = iteration + 1
        #if i_batch >= 10:
        #    break
            
print('Finished..')