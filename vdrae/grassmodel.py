import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
import random
import numpy as np
import pickle
import util
import os


#########################################################################################
## Encoder
#########################################################################################
class LeafEncoder(nn.Module):
    def __init__(self, pc_feature_size, feature_size):
        super(LeafEncoder, self).__init__()
        self.mlp_feature = nn.Linear(pc_feature_size, feature_size)
        self.mlp_box = nn.Linear(3, int(feature_size*0.5))
        self.mlp_label = nn.Linear(1, int(feature_size*0.5))
        self.tanh = nn.Tanh()

    def forward(self, pc_feature, box, label):
        box = self.mlp_box(box[:,0:3])
        box = self.tanh(box)
        label = label.view(-1,1)
        label = self.mlp_label(label)
        label = self.tanh(label)
        feature = torch.cat((box,label), 1)
        return feature

class NonLeafEncoder(nn.Module):
    def __init__(self, pc_feature_size, feature_size):
        super(NonLeafEncoder, self).__init__()
        self.mlp_feature = nn.Linear(pc_feature_size, feature_size)
        self.mlp_box = nn.Linear(3, int(feature_size*0.5))
        self.mlp_label = nn.Linear(1, int(feature_size*0.5))
        self.mlp_left = nn.Linear(feature_size, feature_size)
        self.mlp_right = nn.Linear(feature_size, feature_size)
        self.encoder = nn.Linear(feature_size*2, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, pc_feature, box, label, left, right):
        box = self.mlp_box(box[:,0:3])
        box = self.tanh(box)
        label = label.view(-1,1)
        label = self.mlp_label(label)
        label = self.tanh(label)
        context = self.mlp_left(left)
        context += self.mlp_right(right)
        context = self.tanh(context)
        feature = torch.cat((box,label,context), 1)
        feature = self.encoder(feature)
        feature = self.tanh(feature)
        return feature

class Sampler(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(hidden_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, hidden_size)
        self.mlp2var = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        return input
        """
        encode = self.tanh(self.mlp1(input[:,0:input.shape[1]-6]))
        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)
        std = logvar.mul(0.5).exp_() # calculate the STDEV
        eps = Variable(torch.FloatTensor(std.size()).normal_().cuda()) # random normalized noise
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.cat([eps.mul(std).add_(mu)], 1)
        """

#########################################################################################
## Decoder
#########################################################################################
class NodeDecoder(nn.Module):
    def __init__(self, feature_size):
        super(NodeDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, int(feature_size*0.01))
        self.mlp_left = nn.Linear(feature_size+int(feature_size*0.01), feature_size)
        self.mlp_right = nn.Linear(feature_size+int(feature_size*0.01), feature_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature, left_encode_feature, right_encode_feature):
        parent_feature = self.mlp(parent_feature)
        parent_feature = self.tanh(parent_feature)
        left_feature = torch.cat((parent_feature,left_encode_feature), 1)
        left_feature = self.mlp_left(left_feature)
        left_feature = self.tanh(left_feature)
        right_feature = torch.cat((parent_feature,right_encode_feature), 1)
        right_feature = self.mlp_right(right_feature)
        right_feature = self.tanh(right_feature)
        return left_feature, right_feature

class BoxDecoder(nn.Module):
    def __init__(self, pc_feature_size, feature_size):
        super(BoxDecoder, self).__init__()
        self.mlp1 = nn.Linear(pc_feature_size, feature_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(feature_size, feature_size)
        self.mlp3 = nn.Linear(feature_size+100+feature_size, 3)
        self.softmax = nn.Softmax()
        self.mlp_box = nn.Linear(3, 100)
        self.mlp_feature = nn.Linear(feature_size, feature_size)

    def forward(self, pc_feature, input_box, feature):
        output = self.mlp1(pc_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        output = self.tanh(output)
        output_box = self.mlp_box(input_box)
        output_box = self.tanh(output_box)
        output_feature = self.mlp_feature(feature)
        output_feature = self.tanh(output_feature)

        vector = torch.cat((output, output_box, output_feature), dim=1)
        reg = self.mlp3(vector)
        return reg

class NodeClassifier(nn.Module):
    def __init__(self, pc_feature_size, feature_size):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(pc_feature_size, feature_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(feature_size, feature_size)
        self.mlp3 = nn.Linear(feature_size+100+feature_size, 100)
        self.softmax = nn.Softmax()
        self.mlp_box = nn.Linear(3, 100)
        self.mlp_feature = nn.Linear(feature_size, feature_size)

    def forward(self, pc_feature, input_box, feature):
        output = self.mlp1(pc_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        output = self.tanh(output)
        output_box = self.mlp_box(input_box)
        output_box = self.tanh(output_box)
        output_feature = self.mlp_feature(feature)
        output_feature = self.tanh(output_feature)

        vector = torch.cat((output, output_box, output_feature), dim=1)
        output = self.mlp3(vector)
        return output

class CategoryClassifier(nn.Module):
    def __init__(self, pc_feature_size, feature_size):
        super(CategoryClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, feature_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(feature_size, feature_size)

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        return output

class LeafClassifier(nn.Module):
    def __init__(self, feature_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, feature_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(feature_size, feature_size)
        self.mlp3 = nn.Linear(feature_size, 100)

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        output = self.tanh(output)
        output = self.mlp3(output)
        return output

class SampleDecoder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input_feature):
        output = self.tanh(self.mlp1(input_feature))
        output = self.tanh(self.mlp2(output))
        return output


#########################################################################################
## GRASSEncoderDecoder
#########################################################################################
class GRASSEncoderDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSEncoderDecoder, self).__init__()
        self.leaf_encoder = LeafEncoder(pc_feature_size = config.pc_feature_size, feature_size = config.feature_size)
        self.nonleaf_encoder = NonLeafEncoder(pc_feature_size = config.pc_feature_size, feature_size = config.feature_size)
        self.sample_encoder = Sampler(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_decoder = NodeDecoder(feature_size = config.feature_size)
        self.box_decoder = BoxDecoder(feature_size = config.feature_size, pc_feature_size = config.pc_feature_size)
        self.sample_decoder = SampleDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_classifier = NodeClassifier(pc_feature_size = config.pc_feature_size, feature_size = config.feature_size)
        self.category_classifier = CategoryClassifier(pc_feature_size = config.pc_feature_size, feature_size = config.feature_size)
        self.leaf_classifier = LeafClassifier(feature_size = config.feature_size)
        self.mseLoss = nn.L1Loss(size_average=True)
        self.creLoss = nn.CrossEntropyLoss()

    def leafEncoder(self, pc_feature, box, label):
        return self.leaf_encoder(pc_feature, box, label)

    def nonLeafEncoder(self, pc_feature, box, label, left, right):
        return self.nonleaf_encoder(pc_feature, box, label, left, right)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

    def nodeDecoder(self, parent_feature, left_encode_feature, right_encode_feature):
        return self.node_decoder(parent_feature, left_encode_feature, right_encode_feature)

    def boxDecoder(self, pc_feature, box, feature):
        return self.box_decoder(pc_feature, box, feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def leafClassifier(self, feature):
        return self.leaf_classifier(feature)

    def nodeClassifier(self, pc_feature, box, feature):
        return self.node_classifier(pc_feature, box, feature)

    def categoryClassifier(self, feature):
        return self.category_classifier(feature)

    def boxLossEstimator(self, pred_reg, gt_reg):
        return torch.cat([self.mseLoss(b, gt)[None] for b, gt in zip(pred_reg[:,0:3], gt_reg[:,0:3])], 0)

    def symLossEstimator(self, sym_param, gt_sym_param):
        return torch.cat([self.mseLoss(s, gt).mul(0.5) for s, gt in zip(sym_param, gt_sym_param)], 0)

    def classifyLossEstimator(self, label_vector, gt_label_vector):
        loss = torch.cat([nn.CrossEntropyLoss()(l.unsqueeze(0), gt[None])[None].mul(0.2) for l, gt in zip(label_vector, gt_label_vector)], 0) 
        return loss
    
    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

    def vectorMultipler(self, v):
        return v.mul_(0.001)
    
    def tensor2Node(self, v):
        return v

# forward pass
def encode_decode_structure_fold(fold, tree):
    def encode_node(node):
        if node.is_leaf():
            return fold.add('leafEncoder', node.pc_feature, node.box, node.category_pred)
        elif node.is_internal():
            node.left.encode_feature = encode_node(node.left)
            node.right.encode_feature = encode_node(node.right)
            return fold.add('nonLeafEncoder', node.pc_feature, node.box, node.category_pred, node.left.encode_feature, node.right.encode_feature)
    def sample_encoder(feature):
        return fold.add('sampleEncoder', feature)

    def decode_node_box(node, feature):
        if node.is_leaf():
            node_label = fold.add('nodeClassifier', node.pc_feature, node.box[:,:3], feature)
            node_cls_loss = fold.add('classifyLossEstimator', node_label, node.category)
            if node.category == 0:
                loss = node_cls_loss
            else:
                box_reg = fold.add('boxDecoder', node.pc_feature, node.box[:,:3], feature)
                reg_loss = fold.add('boxLossEstimator', box_reg, node.reg[:,:3])
                reg_loss = fold.add('vectorMultipler', reg_loss)
                loss = fold.add('vectorAdder', node_cls_loss, reg_loss)
            return loss

        elif node.is_internal():
            node.left.decode_feature, node.right.decode_feature = fold.add('nodeDecoder', feature, node.left.encode_feature, node.right.encode_feature).split(2)
            left_loss = decode_node_box(node.left, node.left.decode_feature)
            right_loss = decode_node_box(node.right, node.right.decode_feature)
            loss_left_right = fold.add('vectorAdder', left_loss, right_loss)

            node_label = fold.add('nodeClassifier', node.pc_feature, node.box[:,:3], feature)
            node_cls_loss = fold.add('classifyLossEstimator', node_label, node.category)
            if node.category == 0:
                loss_cur = node_cls_loss
            else:
                box_reg = fold.add('boxDecoder', node.pc_feature, node.box[:,:3], feature)
                reg_loss = fold.add('boxLossEstimator', box_reg, node.reg[:,:3])
                reg_loss = fold.add('vectorMultipler', reg_loss)
                loss_cur = fold.add('vectorAdder', node_cls_loss, reg_loss)
            loss = fold.add('vectorAdder', loss_cur, loss_left_right)
            return loss

    def sample_decoder(feature):
        return fold.add('sampleDecoder', feature)

    feature = encode_node(tree.root)
    feature = sample_encoder(feature)
    feature = sample_decoder(feature)
    loss = decode_node_box(tree.root, feature)
    return loss


#########################################################################################
## GRASSEncoderDecoder Evaluation
#########################################################################################
def encode_decode_structure_eval(model, tree):
    def encode_node(node):
        if node.is_leaf():
            node.box = Variable(node.box.cuda(), requires_grad=False)
            node.pc_feature = Variable(node.pc_feature.cuda(), requires_grad=False)
            node.category_pred = Variable(node.category_pred.cuda(), requires_grad=False)
            return model.leafEncoder(node.pc_feature, node.box, node.category_pred)
        elif node.is_internal():
            node.left.encode_feature = encode_node(node.left)
            node.right.encode_feature = encode_node(node.right)
            node.box = Variable(node.box.cuda(), requires_grad=False)
            node.pc_feature = Variable(node.pc_feature.cuda(), requires_grad=False)
            node.category_pred = Variable(node.category_pred.cuda(), requires_grad=False)
            return model.nonLeafEncoder(node.pc_feature, node.box, node.category_pred, node.left.encode_feature, node.right.encode_feature)
    def encode_node_variable(node):
        if node.is_leaf():
            node.box = Variable(node.box.cuda(), requires_grad=False)
            return model.boxEncoder(node.pc_feature, node.box)
        elif node.is_internal():
            left = encode_node_variable(node.left)
            right = encode_node_variable(node.right)
            node.box = Variable(node.box.cuda(), requires_grad=False)
            node.feature = Variable(node.feature.cuda(), requires_grad=False)
            return model.adjEncoder(left,right, node.feature, node.box)
    def sample_encoder(feature):
        output = model.sampleEncoder(feature)
        return output

    def decode_node_box(node, feature, gts, predictions, obbs, labels, ids, count):
        if node.is_leaf():
            nodel_cls = model.nodeClassifier(node.pc_feature, node.box[:,:3], feature)
            nodel_cls = nn.Softmax()(nodel_cls)
            value, node_label = torch.max(nodel_cls, 1)
 
            gts[count] = node.category
            labels[count] = node_label.cpu()
            predictions[count] = nodel_cls.cpu().data.numpy()[0]
            box_reg = model.boxDecoder(node.pc_feature, node.box[:,:3],feature)
            box = node.box.cpu().data.numpy()[0]
            box_adjust = node.box.cpu().data.numpy()[0]
            reg = box_reg.cpu().data.numpy()[0]
            box_adjust[3:] = box[3:]
            box_adjust[:3] = box[:3] + reg[:3]
            obbs[count] = box_adjust
            ids[count] = node.index
            count = count + 1

            return gts, predictions, obbs, labels, ids, count
        elif node.is_internal():
            nodel_cls = model.nodeClassifier(node.pc_feature, node.box[:,:3], feature)
            nodel_cls = nn.Softmax()(nodel_cls)
            value, node_label = torch.max(nodel_cls, 1)
 
            gts[count] = node.category
            labels[count] = node_label.cpu()
            predictions[count] = nodel_cls.cpu().data.numpy()[0]
            box_reg = model.boxDecoder(node.pc_feature, node.box[:,:3],feature)
            box = node.box.cpu().data.numpy()[0]
            box_adjust = node.box.cpu().data.numpy()[0]
            reg = node.reg.data.numpy()[0]
            box_adjust[3:] = box[3:]
            box_adjust[:3] = box[:3] + reg[:3]
            obbs[count] = box_adjust
            ids[count] = node.index
            count = count + 1

            left, right = model.nodeDecoder(feature, node.left.encode_feature, node.right.encode_feature)
            left_gts, left_predictions, obbs, labels, ids, count = decode_node_box(node.left, left, gts, predictions, obbs, labels, ids, count)
            right_gts, right_predictions, obbs, labels, ids, count = decode_node_box(node.right, right, gts, predictions, obbs, labels, ids, count)
            
            return gts, predictions, obbs, labels, ids, count

    def sample_decoder(feature):
        output = model.sampleDecoder(feature)
        return output

    feature1 = encode_node(tree.root)
    feature2 = sample_encoder(feature1)
    feature3 = sample_decoder(feature2)
    gts = [None]*100000
    predictions = [None]*100000
    obbs = [None]*10000
    labels = [None]*10000
    ids = [None]*10000
    count = 0
    gts, predictions, obbs, labels, ids, count = decode_node_box(tree.root, feature3, gts, predictions, obbs, labels, ids, count)
    gts, predictions, obbs, labels, ids = gts[0:count], predictions[0:count], obbs[0:count], labels[0:count],ids[0:count]
    return gts, predictions, obbs, labels, ids, count