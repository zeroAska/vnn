import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature
from egnn_pytorch import EGNN
import ipdb

class get_model(nn.Module):
    def __init__(self, args, num_part=50, normal_channel=False):
        super(get_model, self).__init__()

        self.args = args
        self.n_knn = args.n_knn
        self.num_part = num_part
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        #'''
        self.conv1 = VNLinearLeakyReLU(1, 64//3) #VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3, 64//3) #VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3, 64//3) #VNLinearLeakyReLU(64//3*2, 64//3)
        #'''
        self.egnn1 = EGNN(dim=64//3, num_nearest_neighbors=self.n_knn)
        self.egnn2 = EGNN(dim=64//3, num_nearest_neighbors=self.n_knn)
        self.egnn3 = EGNN(dim=64//3, num_nearest_neighbors=self.n_knn)
        self.egnn4 = EGNN(dim=64//3, num_nearest_neighbors=self.n_knn)
        self.egnn5 = EGNN(dim=64//3, num_nearest_neighbors=self.n_knn)
        self.egnn6 = EGNN(dim=64//3, num_nearest_neighbors=self.n_knn)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 1024//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        #self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
        self.conv8 = nn.Sequential(nn.Conv1d(88, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)
        

    def forward(self, x, l, feat=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        #x = x.unsqueeze(1) # torch.Size([16, 1, 3, 2048])
        x = torch.permute(x, (0 , 2, 1))
        if feat is None:
            feat = torch.ones((batch_size, num_points, 21), device='cuda')
        

        #x = get_graph_feature(x, k=self.n_knn, is_dir_only=True) # torch.Size([16, 2, 3, 2048, 40])
        #x = self.conv1(x) # torch.Size([16, 21, 3, 2048, 40])
        feat, x = self.egnn1(feat, x) # torch.Size([16, 21, 3, 2048, 40])
        feat, x = self.egnn2(feat, x) # torch.Size([16, 21, 3, 2048, 40])
        feat1, x1 = feat, x
        
        #x = self.conv2(x) # torch.Size([16, 21, 3, 2048, 40])
        #x1 = self.pool1(x) # torch.Size([16, 21, 3, 2048]) # pool from the vn vec dimension
        
        #x = get_graph_feature(x1, k=self.n_knn, is_dir_only=True) # torch.Size([16, 42, 3, 2048, 40])
        #x = self.conv3(x) # torch.Size([16, 21, 3, 2048, 40])
        #x = self.conv4(x)
        #x2 = self.pool2(x)# torch.Size([16, 21, 3, 2048])
        feat, x = self.egnn3(feat, x) # torch.Size([16, 21, 3, 2048, 40])
        feat, x = self.egnn4(feat, x) # torch.Size([16, 21, 3, 2048, 40])
        feat2, x2 = feat, x
        
        
        #x = get_graph_feature(x2, k=self.n_knn, is_dir_only=True)
        #x = self.conv5(x)
        #x3 = self.pool3(x) # torch.Size([16, 21, 3, 2048])
        feat, x = self.egnn5(feat, x) # torch.Size([16, 21, 3, 2048, 40])
        feat3, x3 = feat, x
        
        feat123 = feat #torch.cat((feat1, feat3), dim=1)
        x123 = x #torch.cat((x1, x3), dim=1) # torch.Size([16, 63, 3, 2048])
        
        

        feat, x = self.egnn6(feat123, x123)

        x = torch.permute(feat, (0, 2, 1))
        #x = self.conv6(x123) # torch.Size([16, 341, 3, 2048])

        #x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())  # torch.Size([16, 341, 3, 2048])
        #x = torch.cat((x, x_mean), 1)
        #x, z0 = self.std_feature(x) # x: torch.Size([16, 682, 3, 2048]), z0: torch.Size([16, 3, 3, 2048])

        # matrix multiplication
        #x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points) # torch.Size([16, 189, 2048])
        #x = x.view(batch_size, -1, num_points) # torch.Size([16, 2046, 2048])
        x = x.max(dim=-1, keepdim=True)[0] # torch.Size([16, 2046, 1])

        ## Starting from conv7, the nn are linear layers instead of vnn
        l = l.view(batch_size, -1, 1) # torch.Size([16, 16, 1])
        l = self.conv7(l) # torch.Size([16, 64, 1])

        x = torch.cat((x, l), dim=1) # torch.Size([16, 2110, 1])
        x = x.repeat(1, 1, num_points) # torch.Size([16, 2110, 2048])

        x123 = torch.permute(x123, (0,2,1))
        x = torch.cat((x, x123), dim=1) # torch.Size([16, 2299, 2048])

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)
        
        trans_feat = None
        return x.transpose(1, 2), trans_feat


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')
            
        return loss
