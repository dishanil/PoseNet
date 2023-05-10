import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
import PoseNet
from PoseNet import posenet_v1, PoseLoss
from DataSet import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math

from flopth import flopth

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
   
class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        
        conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        # x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.og_model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

        self.model = torch.nn.Sequential(*(list(self.og_model.children())[:-1]))

        self.aux1_xyz = InceptionAux(512, 3, dropout=0.7)
        self.aux1_wpqr = InceptionAux(512, 4, dropout=0.7)

        self.aux2_xyz = InceptionAux(528, 3, dropout=0.7)
        self.aux2_wpqr = InceptionAux(528, 4, dropout=0.7)

        self.cls_fc_pose_xyz = nn.Linear(1024, 3)
        self.cls_fc_pose_wpqr = nn.Linear(1024, 4)

        self.model_1 = torch.nn.Sequential(self.model, self.cls_fc_pose_xyz)
        self.model_2 = torch.nn.Sequential(self.model, self.cls_fc_pose_wpqr)

    def forward(self, x):
        # TODO return unnormalized log-probabilities here
        self.og_model.inception4a.register_forward_hook(get_activation('inception4a'))
        self.og_model.inception4d.register_forward_hook(get_activation('inception4d'))

        pre_output = self.model(x).squeeze()

        cls_xyz = self.cls_fc_pose_xyz(pre_output)
        cls_wpqr = self.cls_fc_pose_wpqr(pre_output)

        output_4a = activation['inception4a']

        cls_inter_1_xyz = self.aux1_xyz(output_4a)
        cls_inter_1_wpqr = self.aux1_wpqr(output_4a)

        output_4d = activation['inception4d']

        cls_inter_2_xyz = self.aux2_xyz(output_4d)
        cls_inter_2_wpqr = self.aux2_wpqr(output_4d)

        return cls_inter_1_xyz, cls_inter_1_wpqr, cls_inter_2_xyz, cls_inter_2_wpqr, cls_xyz, cls_wpqr


import torch
import torch.nn as nn

def qconj(q):
    """
    Conjugate (inverse) of quaternion
    """
    assert q.shape[-1] == 4

    conj = q.clone()
    conj[..., -3:] *= -1
    return conj


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3] - terms[:, 3, 2]
    y = terms[:, 0, 2] - terms[:, 1, 3] + terms[:, 2, 0] + terms[:, 3, 1]
    z = terms[:, 0, 3] + terms[:, 1, 2] - terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def sinc(x):
    return x.sin() / x

def boxplus(quat, delta):
    corr_delta = delta / 2
    delta_norm = torch.norm(corr_delta, 2, dim=-1, keepdim=True)
    exp_delta = torch.cat((delta_norm.cosh(), sinc(delta_norm) * corr_delta),
                          -1)
    exp_delta /= torch.norm(exp_delta, 2, dim=-1, keepdim=True)
    p = qmul(quat, exp_delta)
    p /= torch.norm(p, dim=-1, keepdim=True)
    return p.squeeze()


def boxminus(quat1, quat2):
    inner = qmul(qconj(quat2), quat1)
    v: torch.Tensor = inner[..., 1:]
    w: torch.Tensor = inner[..., :1]
    norm_v = torch.norm(v, 2, dim=-1, keepdim=True)

    log1 = torch.where(
        torch.stack(
            [torch.logical_and((w == 0).squeeze(),
                               (norm_v != 0).squeeze())] * 3, -1),
        (3.14159 * v) / (2 * norm_v), torch.zeros_like(v))

    log2 = torch.where(
        torch.stack(
            [torch.logical_and((norm_v != 0).squeeze(),
                               (w != 0).squeeze())] * 3, -1),
        torch.atan(norm_v / w) * v / norm_v, torch.zeros_like(v))

    return 2 * (log1 + log2)


def qrot(vector, rot):
    """
    Rotate vector(s) about the rotation described by quaternion(s) rot.
    Expects a tensor of shape (*, 4) for rot and a tensor of shape
    (*, 3) for vector, where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert rot.shape[-1] == 4
    assert vector.shape[-1] == 3
    assert rot.shape[:-1] == vector.shape[:-1]

    original_shape = vector.shape
    rot = rot.view(-1, 4)
    vector = vector.view(-1, 3)

    qvec = rot[:, 1:]
    uv = torch.cross(qvec, vector, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (vector + 2 * (rot[:, :1] * uv + uuv)).view(original_shape)


def grad(layer, requires: bool):
    for param in layer.parameters():
        param.requires_grad = requires


class EKF(nn.Module):
    def __init__(self):
        super(EKF, self).__init__()
        self._xHat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float)
        self._p = torch.eye(3) * 1e-10
        self._Q = torch.eye(3) * 1e-10
        self._R = torch.eye(3) * 0.0005
        self._A = torch.eye(4) * 1.0
        self._B = torch.empty((4, 3))
        self._K = torch.empty((3, 3))
        self._dt = torch.tensor(0.01)
        self.reset_filter()

    def to(self, device):
        new_self = super().to(device)
        for k, v in vars(new_self).items():
            if callable(getattr(v, "to", None)):
                setattr(new_self, k, v.to(device))
        return new_self

    def reset_filter(self):
        self.xHat = self._xHat
        self.p = self._p
        self.Q = self._Q
        self.R = self._R
        self.A = self._A
        self.B = self._B
        self.K = self._K
        self.dt = self._dt

    def filter_predict(self, x):
        q = self.xHat
        idx = torch.tensor(
            [[1, 2, 3], [0, 3, 2], [3, 0, 1], [2, 1, 0]],
            dtype=torch.long,
            device=q.device,
        )
        sign = torch.tensor(
            [[-1, -1, -1], [1, -1, 1], [1, 1, -1], [-1, 1, 1]],
            dtype=torch.float,
            device=q.device,
        )
        Sq = torch.gather(q, 0, idx.reshape(-1)).reshape(idx.shape) * sign

        self.B = self.dt / 2 * Sq
        self.xHat = self.A @ self.xHat + self.B @ x[3:6].T
        self.xHat /= self.xHat.norm()
        self.p += self.Q

    def filter_update(self, pred):
        z = pred[:4]
        y = boxminus(z, self.xHat)
        S = self.p + self.R
        self.K = self.p @ S.inverse()
        self.xHat = boxplus(self.xHat, self.K @ y)
        self.xHat /= self.xHat.norm()
        self.p = (torch.eye(3, device=self.K.device) - self.K) @ self.p

    def forward(self, x, pred):
        self.xHat = pred[0, 0, :4]
        states = []
        states.append(torch.cat((self.xHat, self.p.flatten())))
        for i in range(1, x.shape[1]):
            self.filter_predict(x[0, i - 1])
            self.filter_update(pred[0, i])
            states.append(torch.cat((self.xHat, self.p.flatten())))
        states = torch.stack(states, 0)
        return states[None, ...]


class IDOL(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float, fc1_size: int, with_ekf: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc1_size = fc1_size

        self.orient_lstm = nn.LSTM(
            input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True, dropout=self.dropout
        )
        self.orient_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.fc1_size), nn.Tanh(
            ), nn.Linear(self.fc1_size, 4 + 6)
        )

        self.hidden = None
        self.with_ekf = with_ekf
        self.ekf = EKF()

    def to(self, device):
        new_self = super().to(device)
        new_self.ekf = new_self.ekf.to(device)
        return new_self

    def init_hidden(self, batch_sz, device):
        self.hidden = (torch.zeros((self.num_layers, batch_sz, self.hidden_size), device=device),
                       torch.zeros((self.num_layers, batch_sz, self.hidden_size), device=device))

    def forward(self, x):
        orient_out, _ = self.orient_lstm(x, self.hidden)
        orient_out = self.orient_fc(orient_out)
        orient_out = orient_out / \
            torch.norm(orient_out, 2, dim=-1, keepdim=True)
        if self.with_ekf:
            orient_out = self.ekf(x, orient_out)

        return orient_out


class IDOLPos(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        """
        Takes global device orientation as quaternion and raw IMU signal (acc
        & gyro, XYZ order in world frame).  Returns rotated IMU signal with
        gravitational acceleration removed. Magnetometer doesn't need rotation.
        Rotation correctness validation is in magnetic_map.ipynb
        """
        self.frame_rot = nn.Parameter(  # type: ignore
            torch.FloatTensor([0, 0, 9.81645]), requires_grad=False
        )
        self.pos_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=100,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.pos_fc = nn.Sequential(
            nn.Linear(200, 50), nn.Tanh(), nn.Linear(50, output_size)
        )
        self.hidden = None

    def init_hidden(self, batch_sz, device):
        self.hidden = (torch.zeros((4, batch_sz, 100), device=device),
                       torch.zeros((4, batch_sz, 100), device=device))

    def rotate(self, orientation, imu):
        print(orientation.shape)
        acc = qrot(imu[..., :3], orientation)
        acc = (acc.view(-1, 3) - self.frame_rot).view(acc.shape)
        gyro = qrot(imu[..., 3:6], orientation)
        return torch.cat([acc, gyro], dim=-1)

    def forward(self, inp):
        imu = inp[..., :6]
        print("imu.shape = ", imu.shape)
        orientation = inp[..., 9:13]
        orientation = self.rotate(orientation, imu)
        out, _ = self.pos_lstm(orientation, self.hidden)
        out = self.pos_fc(out)
        #out = torch.cumsum(out, dim=1)

        return out


# my_model = MyModel()
# flops, params = flopth(my_model, in_size=((3, 256, 256),))
# print(flops, params)

my_model = IDOLPos(6, 3)
flops_pos, params_pos = flopth(my_model, in_size=((1300000, 13),))
print(flops_pos, params_pos)

my_model = IDOL()
flops_idol, params_idol = flopth(my_model, in_size=((1300000, 9),))

print(flops_idol, params_idol)
