import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from dgl.nn.pytorch import GraphConv
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair

class BertClassifier(torch.nn.Module):
    def __init__(self, pretrained_model:str ="klue/roberta-base", nb_class:int = 8):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(torch.nn.Module):
    def __init__(self, pretrained_model:str ="klue/roberta-base", nb_class:int = 8, ratio:float = 0.7, gcn_layers:int = 2, n_hidden:int = 200, dropout:int = 0.5):
        super(BertGCN, self).__init__()
        self.ratio = ratio
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.ratio + cls_pred * (1 - self.ratio)
        pred = torch.log(pred)
        return pred

class GCN(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GraphConvEdgeWeight(in_feats, n_hidden, activation=activation, norm=normalization))
        for i in range(n_layers - 1):
            self.layers.append(GraphConvEdgeWeight(n_hidden, n_hidden, activation=activation, norm=normalization))
        self.layers.append(GraphConvEdgeWeight(n_hidden, n_classes, norm=normalization))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, features, g, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)
        return h
    
class GraphConvEdgeWeight(GraphConv):

    def forward(self, graph, feat,  weight=None, edge_weights=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst