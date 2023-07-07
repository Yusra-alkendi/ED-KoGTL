#-------------------------Read Dependencies

#read dependancies
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers



import torch
from torch import nn 
import torch.nn.functional as F
import numpy
import os.path as osp
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, BatchNorm1d as BN, ReLU
 
import dgl
from dgl import function as fn
from dgl.utils import expand_as_pair
import torch
from dgl.nn.pytorch import GraphConv, EdgeConv
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
import os

import time 
import csv
import sys
import random 



seed_val = int(1)
print("Random Seed ID is: ", seed_val)
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
os.environ['PYTHONHASHSEED'] = str(seed_val)

#-------------------------Read dataset



import pickle
## read 
#exp749 good light
file_name749 = "data0808/training_dataset_exp749.pkl"
open_file = open(file_name749, "rb") 
loaded_training_dataset_exp749 = pickle.load(open_file)
open_file.close()
print('loaded_training_dataset_exp749 is uploaded',len(loaded_training_dataset_exp749))



file_name749_2 = "data0808/test_dataset_exp749.pkl"
open_file = open(file_name749_2, "rb") 
loaded_test_dataset_exp749 = pickle.load(open_file)
open_file.close()
print('loaded_test_dataset_exp749 is uploaded',len(loaded_test_dataset_exp749))



file_name749_part0 = "data0808/training_dataset0.pkl"
open_file = open(file_name749_part0, "rb") 
loaded_training_dataset0 = pickle.load(open_file)
open_file.close()
print('loaded_training_dataset0 is uploaded',len(loaded_training_dataset0))




file_name749_2_part0 = "data0808/test_dataset0.pkl"
open_file = open(file_name749_2_part0, "rb") 
loaded_test_dataset0= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset0 is uploaded',len(loaded_test_dataset0))



file_name749_part1 = "data0808/training_dataset1.pkl"
open_file = open(file_name749_part1, "rb") 
loaded_training_dataset1= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset1 is uploaded',len(loaded_training_dataset1))



file_name749_2_part1 = "data0808/test_dataset1.pkl"
open_file = open(file_name749_2_part1, "rb") 
loaded_test_dataset1=  pickle.load(open_file)
open_file.close()
print('loaded_test_dataset1 is uploaded',len(loaded_test_dataset1))


file_name749_part2 = "data0808/training_dataset2.pkl"
open_file = open(file_name749_part2, "rb") 
loaded_training_dataset2= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset2 is uploaded',len(loaded_training_dataset2))

file_name749_2_part2 = "data0808/test_dataset2.pkl"
open_file = open(file_name749_2_part2, "rb") 
loaded_test_dataset2= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset2 is uploaded',len(loaded_test_dataset2))



file_name749_part3 = "data0808/training_dataset3.pkl"
open_file = open(file_name749_part3, "rb") 
loaded_training_dataset3= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset3 is uploaded',len(loaded_training_dataset3))

file_name749_2_part3 = "data0808/test_dataset3.pkl"
open_file = open(file_name749_2_part3, "rb") 
loaded_test_dataset3= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset3 is uploaded',len(loaded_test_dataset3))



file_name749_part4 = "data0808/training_dataset4.pkl"
open_file = open(file_name749_part4, "rb") 
loaded_training_dataset4= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset4 is uploaded',len(loaded_training_dataset4))

file_name749_2_part4 = "data0808/test_dataset4.pkl"
open_file = open(file_name749_2_part4, "rb") 
loaded_test_dataset4= pickle.load(open_file)

open_file.close()
print('loaded_test_dataset4 is uploaded',len(loaded_test_dataset4))


#exp5 low light

file_name5 = "data0808/training_dataset_exp5.pkl"
open_file = open(file_name5, "rb") 
loaded_training_dataset_exp5=pickle.load(open_file)
open_file.close()
print('loaded_training_dataset_exp5 is uploaded',len(loaded_training_dataset_exp5))

file_name5_v2 = "data0808/test_dataset_exp5.pkl"
open_file = open(file_name5_v2, "rb") 
loaded_test_dataset_exp5= pickle.load(open_file)

open_file.close()
print('loaded_test_dataset_exp5 is uploaded',len(loaded_test_dataset_exp5))


#part5

file_name5_part5 = "data0808/training_dataset5.pkl"
open_file = open(file_name5_part5, "rb") 
loaded_training_dataset5= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset5 is uploaded',len(loaded_training_dataset5))

file_name5_2_part5 = "data0808/test_dataset5.pkl"
open_file = open(file_name5_2_part5, "rb") 
loaded_test_dataset5= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset5 is uploaded',len(loaded_test_dataset5))

#part6
file_name5_part6 = "data0808/training_dataset6.pkl"
open_file = open(file_name5_part6, "rb") 
loaded_training_dataset6= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset6 is uploaded',len(loaded_training_dataset6))

file_name5_2_part6 = "data0808/test_dataset6.pkl"
open_file = open(file_name5_2_part6, "rb") 
loaded_test_dataset6= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset6 is uploaded',len(loaded_test_dataset6))


#part7
file_name5_part7 = "data0808/training_dataset7.pkl"
open_file = open(file_name5_part7, "rb") 
loaded_training_dataset7= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset7 is uploaded',len(loaded_training_dataset7))

file_name5_2_part7 = "data0808/test_dataset7.pkl"
open_file = open(file_name5_2_part7, "rb") 
loaded_test_dataset7= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset7 is uploaded',len(loaded_test_dataset7))




#part8
file_name5_part8 = "data0808/training_dataset8.pkl"
open_file = open(file_name5_part8, "rb") 
loaded_training_dataset8= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset8 is uploaded',len(loaded_training_dataset8))
file_name5_2_part8 = "data0808/test_dataset8.pkl"
open_file = open(file_name5_2_part8, "rb") 
loaded_test_dataset8= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset8 is uploaded',len(loaded_test_dataset8))

#part9
file_name749_part4 = "data0808/training_dataset9.pkl"
open_file = open(file_name749_part4, "rb") 
loaded_training_dataset9= pickle.load(open_file)
open_file.close()
print('loaded_training_dataset9 is uploaded',len(loaded_training_dataset9))

file_name5_2_part9 = "data0808/test_dataset9.pkl"
open_file = open(file_name5_2_part9, "rb") 
loaded_test_dataset9= pickle.load(open_file)
open_file.close()
print('loaded_test_dataset9 is uploaded',len(loaded_test_dataset9))



TrainingSet=loaded_training_dataset_exp749+loaded_training_dataset_exp5
print('TrainingSet', len(TrainingSet))
TestingSet=loaded_test_dataset_exp749+loaded_test_dataset_exp5

print('TestingSet', len(TestingSet))






#-------------------------Network and training

## transformer
import copy
from typing import Optional, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

## creat graphs dataset class
class EventGraphDataset2(object):

    def __init__(self, seq):
        super(EventGraphDataset2, self).__init__()
        self.graphs = []
        self.labels = []
        self.toNode=[]
        self.features=[]
        self.IDdataXYT=[]
        self._generate(seq)

        

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):

        return self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2
        
        
        
    def _generate(self, seq):

        total_samples = 0
        i = 0
        #data=[]
        for sample in range(0, len(seq)):
            #sample=[0]
            i += 1
            total_samples += 1
 
    #######################################tonode of interest from others##################3333

              

    
    #######################################################################################3
          #  full_edgesz = EdgeData_per_sequences1[0, :,:]
            to_nodesz = []
            from_nodesz  = seq[sample].shape[0]-1#full_edgesz[:, 1]          
            for i in range(seq[sample].shape[0]):
                to_nodesz.append(i)

           # print('to_nodesz', to_nodesz)
            #print('from_nodesz', from_nodesz)



            EventDataIDXYTL=seq[sample][:,1:5]#sequences[sample,:, 0:5]


            data = dgl.DGLGraph()
            data.add_nodes(int(seq[sample].shape[0]))
            #print('databeforeedge',data)
           # data.add_edges((torch.tensor(from_nodes, dtype=torch.int64).reshape(10)), (torch.tensor(to_nodes, dtype=torch.int64).reshape(10)))
            #print(data)

            costs= numpy.transpose(numpy.zeros([1,10]))

            bb=EventDataIDXYTL#.reshape(EventDataIDXYTL.shape[1],EventDataIDXYTL.shape[2])
            #print('bb', bb)
            node_features = (bb[:,0:3])#[:, 1:4])

            data.ndata['x'] = torch.tensor(node_features, dtype=torch.float)
            #data.add_edges(from_nodes.astype(int)-sample*int(EdgeData_per_sequences1.shape[1]),to_nodes.astype(int)-sample*int(EdgeData_per_sequences1.shape[1]))
            data.add_edges(from_nodesz,to_nodesz)

    
            #data.edata['y'] = torch.tensor(edges, dtype=torch.float)
            NodeLabel=EventDataIDXYTL[:,3]
            IDdataXYT=seq[sample][from_nodesz,0:4]

            #print('NodeLabel',NodeLabel)

            #print(data)
            self.features.append(data.ndata['x'])
            self.graphs.append(data)
            self.toNode.append(to_nodesz)    
            self.labels.append(int(NodeLabel[from_nodesz]))
            self.IDdataXYT.append(IDdataXYT)



## # Create graphdataset from training/testing sets

print("Reading training Dataset Samples")
TrainingGrphLabl =EventGraphDataset2(TrainingSet)
print("Reading Testing Dataset Samples")
TestingGrphLabl =EventGraphDataset2(TestingSet)



## # Graph Layer and Network Classifier 
#####My custumize Event-conv layer
class E_EdgeConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False):
        super(E_EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self.in_feat = in_feat
        self.out_feat = out_feat
            #self.theta = nn.Dense(out_feat, in_units=in_feat,
             #                     weight_initializer=mx.init.Xavier())
        print('1')
        self.alpha =  nn.Linear(in_feat, in_feat)

            
        if batch_norm:
            self.bn = nn.BatchNorm1d(in_channels=out_feat)

        
        
    def message(self, edges):
        r"""The message computation function
        """
        #print('2')
        #print(edges.dst['x'] - edges.src['x'])
        
        xi = edges.src['x'][:, 0]
        xj = edges.dst['x'][:, 0]

        yi = edges.src['x'][:, 1]
        yj = edges.dst['x'][:, 1]

        ti = edges.src['x'][:, 2]
        tj = edges.dst['x'][:, 2]
 

        Node_data_features = (edges.dst['x'])
        mean_of_all_nodes= sum(Node_data_features)/len(Node_data_features) ### mean(deltaX) mean(delta) mean(y deltat)
        xi_mean=mean_of_all_nodes[0]
        yi_mean=mean_of_all_nodes[1]
        ti_mean=mean_of_all_nodes[2]
        #print('xi_mean',xi_mean)
        #print('yi_mean',yi_mean)
        #print('ti_mean',ti_mean)


####################feature  Delta_XYT ### message 1 2 3
        Delta_XYT = (edges.dst['x'] - edges.src['x']) ### deltaX delta y deltat
        #print('Delta_XYT',Delta_XYT)
        
        
####################feature Ecldn_distance  ### message 7

        Ecldn_distance=torch.sqrt((torch.square(xi-xj)+torch.square(yi-yj)+torch.square(ti-tj)))
        #print('Ecldn_distance', Ecldn_distance)

        
####################feature standard deviation x y t ### message 4 5 6
        Standard_deviation_of_x=torch.sqrt((torch.square(xi-xi_mean))/(len(Node_data_features)))
        Standard_deviation_of_y=torch.sqrt((torch.square(yi-yi_mean))/(len(Node_data_features)))
        Standard_deviation_of_t=torch.sqrt((torch.square(ti-ti_mean))/(len(Node_data_features)))
      


        Standard_deviation_xyt=torch.stack([Standard_deviation_of_x,Standard_deviation_of_y, 
                                            Standard_deviation_of_t],dim=1)       
        InputfeaturesfrmGNN=torch.cat( (Delta_XYT, Standard_deviation_xyt,Ecldn_distance.unsqueeze(1) ) ,1)
        
        
       
        InputfeaturesfrmGNN2=self.alpha(InputfeaturesfrmGNN)
        #print('InputfeaturesfrmGNN ready',InputfeaturesfrmGNN2)

        return {'e': InputfeaturesfrmGNN2} ##+ phi_x}

    def forward(self, g, h):

        with g.local_scope():
            #print('4')

            h_src, h_dst = expand_as_pair(h)
            #print('5')

            g.srcdata['x'] = h_src
            #print('6')

            g.dstdata['x'] = h_dst
            #print('7')
            #print('g.ndata[x]',g.ndata['x'])
            if not self.batch_norm:
                g.update_all(self.message, fn.sum('e', 'x'))
             #   print('g.dstdata[x]',g.dstdata['x'])

            else:
                g.apply_edges(self.message)
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'm'), fn.max('m', 'x'))


            return g.dstdata['x']


class NetClassifier2(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NetClassifier2, self).__init__()
        #print('1')
        self.in_feats=in_feats
        self.layer1 = E_EdgeConv(in_feats,in_feats)
        self.classify = nn.Linear(in_feats+1, out_feats)
        self.hiddenLayer= nn.Linear(1,1)
        self.classify2 = nn.Linear(1, out_feats)
        self.classify3 = nn.Linear(in_feats, out_feats)
        self.hiddenLayer3= nn.Linear(in_feats,in_feats)
        self.Transformer=nn.Transformer(d_model=in_feats, nhead=in_feats, dim_feedforward=2*in_feats ,
                                        num_encoder_layers=1, num_decoder_layers=1)
    def forward(self, g):#, features):
        node_features=g.ndata['x']
        x = F.sigmoid(self.layer1(g,node_features))#, features))
        #print('layer',self.layer1(g, features))
        #print('relu',x)
        #x = F.sigmoid(self.layer2(g, x))
        g.ndata['h'] = x
        hg = dgl.sum_nodes(g, 'h')
        #print('len(x)', len(node_features))

        hg2=hg.view(-1,1)
        #print('hg2', hg2)
        vector= hg2#torch.tensor( [10 , 5 , 20]).view(-1,3)
        min_v = torch.min(vector)
        range_v = torch.max(vector) - min_v
        if range_v > 0:
            normalised = (vector - min_v) / range_v
        else:
            normalised = torch.zeros(vector.size())




  
        src1=normalised.view(-1,1,self.in_feats)
        #print('src', src1)
        tgt=(self.hiddenLayer3(normalised.view(-1,1,self.in_feats)))
       # print('tgt', tgt)

        

        #print('tgt(0)', src1.size(1))
        #print('src(0)', tgt.size(1))

        
        
        
        
        
        ysrc = torch.rand((1, 1,self.in_feats))
        ytgt = torch.rand((1,1,  self.in_feats))

        Transformer_out=self.Transformer(src1, tgt)

        return self.classify3(Transformer_out)











class Transformer(Module):
    r"""

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()
        self.d_model = d_model

        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        print('src.size(1) yusra ', src.size(1) )

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")
        print('T2')

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        print('T3')

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        print('T4')

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        print('T5')

        return output


    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p) 




class TransformerEncoder(Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.


        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".

    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        print('1')
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        print('2')

        src = src + self.dropout1(src2)
        print('3')

        src = self.norm1(src)
        print('4')

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        print('5')

        src = src + self.dropout2(src2)
        print('6')

        src = self.norm2(src)
        print('7')

        return src



class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first']

    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)



## ------------ load training dataset
print("Reading Training Dataset Samples")

Data_loader4trainset = DataLoader(TrainingGrphLabl, batch_size=64,
                             shuffle=True, collate_fn=collate)



## training module 
import time
net = NetClassifier2(7,2) ## My network GNN-driven trasformer 7 messages 2 L of transformer
print(net)



# Create model
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
number_of_epoch=200
i = 1
epoch_losses = []

print("Training will start now")
for epoch in range(number_of_epoch):
    epoch_loss = 0
    start = time.time()
        #for iter, (bg, label) in enumerate(data_loader):
    featuresAppend=[]
    bgAppend=[]
    lossAppend=[]
    for iter, (bg, label) in enumerate(Data_loader4trainset):
        

        prediction =  net(bg)
        #print('prediction0', prediction)
        #print('label',label)
        loss = loss_func(prediction.view(-1,2), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        i = i+1


    epoch_loss /= (iter + 1)
    end = time.time()
    print('Epoch {}, loss {:.4f}'.format(
        epoch, epoch_loss), ' Elapsed time: ', end-start)
    epoch_losses.append(epoch_loss)
net.eval()



#-------------------------Save and Evaluate


folder = 'TrainingResults/network7msg2Ltrans_'+str(number_of_epoch)+'epochs/'
discpt='Network7msg2Ltrans_'+str(number_of_epoch)+'epochs'
if not os.path.isdir(folder):
    os.makedirs(folder)


torch.save(net.state_dict(), folder+'model_weights.pth')


## save loass in csv

with open(folder+'losses.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows([[loss] for loss in epoch_losses])
    csvFile.close()


## loss plot vs Epoch
plt.title('cross entropy '+discpt)
plt.plot(epoch_losses)
plt.savefig(folder+discpt+str(number_of_epoch)+'epochs.png',dpi=300, bbox_inches='tight')
plt.savefig(folder+discpt+str(number_of_epoch)+'epochs.pdf', format='pdf', dpi=1200)
plt.show()