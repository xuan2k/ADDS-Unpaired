from torch import nn
from torch.autograd import Function


# from the authors of https://github.com/Yangyangii/DANN-pytorch
class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.alpha = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha

        return output, None


class FeatureClassifier(nn.Module):
    def __init__(self, in_channel=512, width=16, height=8, batch_size=8):
        super(FeatureClassifier, self).__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.down = nn.Conv2d(in_channel, 128, 5, 2, padding=2, bias=False)
        self.relu0 = nn.ReLU(True)
        self.fc1 = nn.Linear(int((128 * (width / 8) * (height / 8))), 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(64, 1)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, feature, lambda_=1):
        feature = GradientReversalLayer.apply(feature, lambda_)

        feat = self.down(feature)
        feat = self.relu0(feat)
        feat = feat.view(-1, int(128 * (self.width / 8) * (self.height / 8)))
        feat = self.fc1(feat)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.fc2(feat)
        domain_output = self.soft(feat)

        return domain_output