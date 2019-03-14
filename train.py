import os
import time
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from torchfoldext import FoldExt
from grassdata import GRASSDataset
from grassmodel import GRASSEncoderDecoder
import grassmodel
import util

config = util.get_args()
config.cuda = not config.no_cuda
if config.gpu < 0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
    print("Using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA.")

#pretrained_model = config.pretrained_model
#encoder_decoder = torch.load(pretrained_model)
encoder_decoder = GRASSEncoderDecoder(config)
if config.cuda:
    encoder_decoder.cuda(config.gpu)

print("Loading data ......")
grass_data = GRASSDataset(config.data_path)
def my_collate(batch):
    return batch
train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)

print("Start training ......")
start = time.time()

if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime())+'_'+config.nick_name)
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

header = '     Time    Epoch     Iteration    Progress(%)       LR      TotalLoss'
print(header)
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,   {:>11.9f},{:>10.20f}'.split(','))

learning_rate = config.lr
for epoch in range(config.epochs):
    encoder_decoder_opt = torch.optim.Adam(encoder_decoder.parameters())
    for batch_idx, batch in enumerate(train_iter):
        enc_fold = FoldExt(cuda=config.cuda)
        enc_dec_fold_nodes = []
        enc_dec_recon_fold_nodes = []
        enc_dec_label_fold_nodes = []
        for example in batch:
            enc_dec_fold_nodes.append(grassmodel.encode_decode_structure_fold(enc_fold, example))
        total_loss = enc_fold.apply(encoder_decoder, [enc_dec_fold_nodes])
        sum_loss = total_loss[0].sum()
        encoder_decoder_opt.zero_grad()
        sum_loss.backward()
        encoder_decoder_opt.step()
        if batch_idx % config.show_log_every == 0:
            print(log_template.format(strftime("%H:%M:%S",time.gmtime(time.time()-start)),
                epoch, config.epochs, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                learning_rate, sum_loss))
    if config.save_snapshot and (epoch+0) % config.save_snapshot_every == 0 :
        print("Saving snapshots of the models ...... ")
        torch.save(encoder_decoder, snapshot_folder+'/encoder_decoder_model_epoch_{}_loss_{:.4f}.pkl'.format(epoch+0, sum_loss.data[0]))
   
print("Saving final models ...... ")
torch.save(encoder_decoder, snapshot_folder+'/encoder_decoder_model.pkl')
print("DONE")
