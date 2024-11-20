import types
import torch
from nn import MuReNNDirect
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
import data


class Plmodel(pl.LightningModule):
    def __init__(self, Q, T, J, lr, scale_factor, **kwargs):
        super(Plmodel, self).__init__()
        self.save_hyperparameters()
        self.T = T
        self.Q = Q
        self.scale_factor = scale_factor
        if isinstance(J, int):
            self.J = range(J)
        elif isinstance(J, list):
            self.J = J
        self.loss = nn.BCELoss()
        self.lr = lr
        self.train_outputs = {'loss': [], 'acc': []}
        self.test_outputs = {'acc': []}
        self.val_outputs = {'loss': [], 'acc': []}
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        def int_or_list(value):
            try:
                return int(value)
            except ValueError:
                return list(map(int, value.split(',')))[:-1]
        def float_list_or_str(value):
            try:
                return float(value)
            except ValueError:
                try:
                    return list(map(float, value.split(',')))
                except ValueError:
                    return value
        parser = parent_parser.add_argument_group("model")
        parser.add_argument('--Q', type=int, default=6)
        parser.add_argument('--T', type=int, default=4)
        parser.add_argument('--J', type=int_or_list, default=6, help='Index of octaves to be trained, if list, end by -1 (e.g. 1,-1); if int, all octaves from 0 to J-1 will be trained')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--scale_factor', type=float_list_or_str, default=1)

        return parent_parser
    
    def step(self, batch, fold):
        x, label = batch
        logits = self(x).reshape(label.shape)
        loss = self.loss(logits, label)
        acc = ((logits > 0.5).float() == label).float().mean()
        if fold == 'train':
            self.train_outputs['loss'].append(loss)
            self.train_outputs['acc'].append(acc)
        elif fold == 'val':
            self.val_outputs['loss'].append(loss)
            self.val_outputs['acc'].append(acc)
        elif fold == 'test':
            self.test_outputs['acc'].append(acc)
        return loss
    
    def forward(self, x):
        return self(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)#, weight_decay=1e-3)

    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_outputs['loss']).mean()
        avg_acc = torch.stack(self.train_outputs['acc']).mean()
        self.log('train_loss', avg_loss)
        self.log('train_acc', avg_acc)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_outputs['loss']).mean()
        avg_acc = torch.stack(self.val_outputs['acc']).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
    
    def on_test_epoch_end(self):
        avg_acc = torch.stack(self.test_outputs['acc']).mean()
        self.log('test_acc', avg_acc)
    

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.normal_(m.weight, 0, 1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MuReNN(Plmodel):
    def __init__(self, **kwargs):
        super(MuReNN, self).__init__(**kwargs)
        Jmax = max(self.J) + 1
        self.S1 = MuReNNDirect(in_channels=1, Q=self.Q, T=self.T, J=Jmax, scale_factor=self.scale_factor)
        self.S2 = MuReNNDirect(in_channels=Jmax, Q=self.Q, T=1, J=Jmax//2, scale_factor='in')
        self.classifier = nn.Sequential(
            nn.Linear(Jmax//2 * self.Q * Jmax, 1),
            nn.Sigmoid(),
        )
        self.pointwiseconv = nn.Conv1d(
            in_channels=self.Q*len(self.J),
            out_channels=Jmax,
            kernel_size=1,
            bias=True,
        )
        self.apply(initialize_weights)
        self.slide = [q+self.Q*j for j in self.J for q in range(self.Q)]

    def forward(self, x):
        s = self.S1(x).squeeze(1)
        s  = s[:, self.slide, :]
        s = self.pointwiseconv(s)
        s = self.S2(s)
        s = s.reshape(s.shape[0], -1, s.shape[-1])
        logits = self.classifier(s.mean(dim=-1).reshape(s.shape[0], -1))
        return logits
    
    def on_before_optimizer_step(self, optimizer):
        norm1 = grad_norm(self.S1.conv1d, norm_type=2)
        norm2 = grad_norm(self.S2.conv1d, norm_type=2)
        self.log_dict(norm1, logger=True)
        self.log_dict(norm2, logger=True)


class Conv1D(Plmodel):
    def __init__(self, **kwargs):
        super(Conv1D, self).__init__(**kwargs)
        self.conv1ds = nn.ParameterList(
            [nn.Sequential(
                nn.Conv1d(1, self.Q, kernel_size=self.T, dilation=2**(j+1), padding='same'),
                nn.ReLU(),
            ) for j in self.J]
        )

        # self.pool = nn.MaxPool1d(kernel_size=2**(max(self.J) + 1))
        self.classifier = nn.Sequential(
            nn.Linear(len(self.J) * self.Q, 1),
            nn.Sigmoid(),
        )
        self.apply(initialize_weights)

    def forward(self, x):
        xjs = []
        for j, conv1d in enumerate(self.conv1ds):
            xj = conv1d(x)
            xjs.append(xj)
        xj = torch.cat(xjs, dim=1) 
        # xj = self.pool(xj)
        logits = self.classifier(xj.mean(dim=-1).reshape(xj.shape[0], -1))
        return logits
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.conv1ds, norm_type=2)
        self.log_dict(norms, logger=True)

class WaveNetConv(nn.Module):
    def __init__(self, num_features_in, num_features_out, res_features, filter_len, dilation):
        self.filter_len = filter_len
        super(WaveNetConv, self).__init__()
        self.conv_tanh = nn.Sequential(*[nn.Conv1d(num_features_in, num_features_out, filter_len, dilation=dilation),
                                         nn.Tanh()])
        self.conv_sig = nn.Sequential(*[nn.Conv1d(num_features_in, num_features_out, filter_len, dilation=dilation),
                                        nn.Sigmoid()])
        self.conv_lin = nn.Conv1d(num_features_out, num_features_out, 1, dilation=dilation)
        self.conv_res = nn.Conv1d(num_features_out, res_features, 1, dilation=dilation)
        self.norm = nn.BatchNorm1d(num_features_in)

    def forward(self, x, res):
        '''
        :param x: [batch,  features, timesteps,]
        '''
        x = self.norm(x)
        x_ = self.conv_tanh(x) * self.conv_sig(x)
        x_res = self.conv_res(x_)
        x_ = self.conv_lin(x_)
        if x_.shape[-1] != x.shape[-1]:
            padding_left = int((x.shape[-1] - x_.shape[-1]) // 2)
            padding_right = x.shape[-1] - x_.shape[-1] - padding_left
            x_ = x[:, :, padding_left:-padding_right] + x_
            res = res[:, :, padding_left:-padding_right] + x_res
        else:
            x_ = x + x_
            res = res + x_res
        return x_, res


class WaveNet(Plmodel):
    def __init__(self, **kwargs):
        super(WaveNet, self).__init__(**kwargs)
        self.convs = nn.ModuleList([WaveNetConv(self.Q, self.Q, self.Q, self.T, 2**(j+1))
                                    for j in self.J])
        self.classifer = nn.Sequential(
            nn.Linear(self.Q, 1),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x):
        x = x.repeat(1, self.Q, 1)
        res = x.new_zeros((x.shape[0], self.Q, x.shape[-1]))
        for idx, conv in enumerate(self.convs):
            x, res = conv(x, res)
        logits = self.classifer(res.mean(dim=-1).reshape(res.shape[0], -1))
        return logits