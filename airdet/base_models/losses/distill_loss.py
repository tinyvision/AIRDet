import torch
import torch.nn as nn

class FeatureLoss(nn.Module):
    def __init__(self, distiller, FPN_Channels, loss_weight=0.1):
        super(FeatureLoss, self).__init__()

        if(distiller == "mimic"):
            self.feature_loss = MimicLoss(FPN_Channels, loss_weight)
        elif(distiller == "mgd"):
            self.feature_loss = MGDLoss(FPN_Channels, loss_weight)
        else:
            assert False

    def forward(self, y_s, y_t):
        loss = self.feature_loss(y_s, y_t)

        return loss


class MimicLoss(nn.Module):
    def __init__(self, FPN_Channels, loss_weight=0.1):
        super(MimicLoss, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

        self.align_module = [nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0).to(device) for channel in FPN_Channels]

    def forward(self, y_s, y_t):
        if isinstance(y_s, (tuple, list)):
            assert len(y_s) == len(y_t)
            losses = []
            for idx, (s, t) in enumerate(zip(y_s, y_t)):
                assert s.shape == t.shape
                s = self.align_module[idx](s)
                losses.append(self.mse(s, t))
            loss = sum(losses)
        else:
            assert y_s.shape == y_t.shape
            loss = self.mse(y_s, y_t)
        return self.loss_weight * loss


class MGDLoss(nn.Module):
    def __init__(self, FPN_Channels, loss_weight=0.1, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_weight = loss_weight
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.align_module = [nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0).to(device) for channel in FPN_Channels]

        self.generation = [nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel in FPN_Channels] 

    def forward(self, y_s, y_t):
        if isinstance(y_s, (tuple, list)):
            assert len(y_s) == len(y_t)
            losses = []
            for idx, (s, t) in enumerate(zip(y_s, y_t)):
                assert s.shape == t.shape
                s = self.align_module[idx](s)
                losses.append(self.get_dis_loss(s, t, idx)*self.alpha_mgd)
            loss = sum(losses)
        else:
            assert y_s.shape == y_t.shape
            loss = self.mse(y_s, y_t)
        return self.loss_weight * loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss