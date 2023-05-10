import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

__all__ = ['PoseNet', 'posenet_v1', 'PoseLoss']

class InceptionV1(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionV1, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


# PoseNet
class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, 0.0001, 0.75),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.LocalResponseNorm(5, 0.0001, 0.75),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.a3 = InceptionV1(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionV1(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = InceptionV1(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionV1(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionV1(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionV1(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionV1(528, 256, 160, 320, 32, 128, 128)

        self.a5 = InceptionV1(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionV1(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avg_pool5x5 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv1x12 = nn.Conv2d(528, 128, kernel_size=1, stride=1)
        self.fc = nn.Linear(1024, 2048)
        self.fc2048 = nn.Linear(2048, 1024)

        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()
        self.cls_fc_pose_xyz = nn.Linear(2048, 3)
        self.cls_fc_pose_wpqr = nn.Linear(2048, 4)
        self.cls_fc_pose_xyz_1024 = nn.Linear(1024, 3)
        self.cls_fc_pose_wpqr_1024 = nn.Linear(1024, 4)

    def forward(self, x):
        out = self.pre_layers(x)

        out = self.a3(out)
        out = self.b3(out)

        out = self.max_pool(out)

        out = self.a4(out)
        cls1_pool = self.avg_pool5x5(out)
        cls1_reduction = self.conv1x1(cls1_pool)
        cls1_reduction = F.relu(cls1_reduction)
        cls1_reduction = cls1_reduction.view(cls1_reduction.size(0), -1)
        cls1_fc1 = self.fc2048(cls1_reduction)
        cls1_fc1 = self.relu(cls1_fc1)
        cls1_fc1 = self.dropout7(cls1_fc1)
        cls1_fc_pose_xyz = self.cls_fc_pose_xyz_1024(cls1_fc1)
        cls1_pose_wpqr = self.cls_fc_pose_wpqr_1024(cls1_fc1)

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        cls2_pool = self.avg_pool5x5(out)
        cls2_reduction = self.conv1x12(cls2_pool)
        cls2_reduction = F.relu(cls2_reduction)
        cls2_reduction = cls2_reduction.view(cls2_reduction.size(0), -1)
        cls2_fc1 = self.fc2048(cls2_reduction)
        cls2_fc1 = self.relu(cls2_fc1)
        cls2_fc1 = self.dropout7(cls2_fc1)
        cls2_fc_pose_xyz = self.cls_fc_pose_xyz_1024(cls2_fc1)
        cls2_pose_wpqr = self.cls_fc_pose_wpqr_1024(cls2_fc1)
        out = self.e4(out)

        out = self.max_pool(out)

        out = self.a5(out)
        out = self.b5(out)
        cls3_pool = self.avg_pool(out)
        cls3_pool = cls3_pool.view(cls3_pool.size(0), -1)
        cls3_fc1 = self.fc(cls3_pool)
        cls3_fc1 = self.relu(cls3_fc1)
        cls3_fc1 = self.dropout5(cls3_fc1)
        cls3_fc_pose_xyz = self.cls_fc_pose_xyz(cls3_fc1)
        cls3_pose_wpqr = self.cls_fc_pose_wpqr(cls3_fc1)

        return cls1_fc_pose_xyz, \
               cls1_pose_wpqr, \
               cls2_fc_pose_xyz, \
               cls2_pose_wpqr, \
               cls3_fc_pose_xyz, \
               cls3_pose_wpqr

# class EuclideanDistance(nn.Cell):
#     """calculate euclidean distance"""
#     def __init__(self):
#         super(EuclideanDistance, self).__init__()
#         self.sub = P.Sub()
#         self.mul = P.Mul()
#         self.reduce_sum = P.ReduceSum()
#         self.sqrt = P.Sqrt()

#     def construct(self, predicted, real):
#         res = self.sub(predicted, real)
#         res = self.mul(res, res)
#         res = self.reduce_sum(res, 0)
#         res = self.sqrt(res)
#         res = self.mul(res, res)
#         res = self.reduce_sum(res, 0)
#         res = self.sqrt(res)

#         return res

class PoseLoss(nn.Module):

    def __init__(self, w1_x, w2_x, w3_x, w1_q, w2_q, w3_q):
        super(PoseLoss, self).__init__()
        self.w1_x = w1_x
        self.w2_x = w2_x
        self.w3_x = w3_x
        self.w1_q = w1_q
        self.w2_q = w2_q
        self.w3_q = w3_q
        self.ed = nn.MSELoss()

    def forward(self, p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, poseGT):
        # pose_x = poseGT[:, 0:3]
        # pose_q = poseGT[:, 3:]

        pose_q = poseGT[:, 0:4]
        pose_x = poseGT[:, 4:7]

        pose_q = pose_q / torch.linalg.norm(pose_q)
        p1_q = p1_q / torch.linalg.norm(p1_q)
        p2_q = p2_q / torch.linalg.norm(p2_q)
        p3_q = p3_q / torch.linalg.norm(p3_q)

        # l1_x = self.ed(pose_x, p1_x) * self.w1_x
        # l1_q = self.ed(pose_q, p1_q) * self.w1_q
        # l2_x = self.ed(pose_x, p2_x) * self.w2_x
        # l2_q = self.ed(pose_q, p2_q) * self.w2_q
        # l3_x = self.ed(pose_x, p3_x) * self.w3_x
        # l3_q = self.ed(pose_q, p3_q) * self.w3_q

        # loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q

        # w1_x_x, w1_x_y, w1_x_z, w2_x_x, w2_x_y, w2_x_z, w3_x_x, w3_x_y, w3_x_z = 5, 3, 20, 5, 3, 20, 10, 5, 80 #experiment2
        # w1_x_x, w1_x_y, w1_x_z, w2_x_x, w2_x_y, w2_x_z, w3_x_x, w3_x_y, w3_x_z = 3, 3, 20, 3, 3, 20, 10, 10, 100
        # w1_q_w, w1_q_p, w1_q_q, w1_q_r, w2_q_w, w2_q_p, w2_q_q, w2_q_r, w3_q_w, w3_q_p, w3_q_q, w3_q_r = 150, 150, 150, 

        pose_x_x, pose_x_y, pose_x_z = pose_x[:, 0], pose_x[:, 1], pose_x[:, 2]
        p1_x_x, p1_x_y, p1_x_z = p1_x[:, 0], p1_x[:, 1], p1_x[:, 2]
        p2_x_x, p2_x_y, p2_x_z = p2_x[:, 0], p2_x[:, 1], p2_x[:, 2]
        p3_x_x, p3_x_y, p3_x_z = p3_x[:, 0], p3_x[:, 1], p3_x[:, 2]

        # pose_q_w, pose_q_p, pose_q_q, pose_q_r = pose_q[:, 0], pose_q[:, 1], pose_q[:, 2], pose_q[:, 3]
        # p1_q_w, p1_q_p, p1_q_q, p1_q_r = p1_q[:, 0], p1_q[:, 1], p1_q[:, 2], p1_q[:, 3]
        # p2_q_w, p2_q_p, p2_q_q, p2_q_r = p2_q[:, 0], p2_q[:, 1], p2_q[:, 2], p2_q[:, 3]
        # p3_q_w, p3_q_p, p3_q_q, p3_q_r = p3_q[:, 0], p3_q[:, 1], p3_q[:, 2], p3_q[:, 3]

        l1_x_x = self.ed(pose_x_x, p1_x_x) * self.w1_x * 3
        l1_x_y = self.ed(pose_x_y, p1_x_y) * self.w1_x * 2
        l1_x_z = self.ed(pose_x_z, p1_x_z) * self.w1_x

        l2_x_x = self.ed(pose_x_x, p2_x_x) * self.w2_x * 3
        l2_x_y = self.ed(pose_x_y, p2_x_y) * self.w2_x * 2
        l2_x_z = self.ed(pose_x_z, p2_x_z) * self.w2_x

        l3_x_x = self.ed(pose_x_x, p3_x_x) * self.w3_x * 3
        l3_x_y = self.ed(pose_x_y, p3_x_y) * self.w3_x * 2
        l3_x_z = self.ed(pose_x_z, p3_x_z) * self.w3_x

        l1_q = self.ed(pose_q, p1_q) * self.w1_q
        l2_q = self.ed(pose_q, p2_q) * self.w2_q
        l3_q = self.ed(pose_q, p3_q) * self.w3_q

        loss = l1_x_x + l1_x_y + l1_x_z + l2_x_x + l2_x_y + l2_x_z + l3_x_x + l3_x_y + l3_x_z + l1_q + l2_q + l3_q

        return loss

class PoseLoss_resnet(nn.Module):

    def __init__(self, w1_x, w2_x, w3_x, w1_q, w2_q, w3_q):
        super(PoseLoss_resnet, self).__init__()
        self.w1_x = w1_x
        self.w2_x = w2_x
        self.w3_x = w3_x
        self.w1_q = w1_q
        self.w2_q = w2_q
        self.w3_q = w3_q
        self.ed = nn.MSELoss()

    def forward(self, p3_x, p3_q, poseGT):
        pose_x = poseGT[:, 0:3]
        pose_q = poseGT[:, 3:]

        pose_q = pose_q / torch.linalg.norm(pose_q)
        p1_q = p1_q / torch.linalg.norm(p1_q)
        p2_q = p2_q / torch.linalg.norm(p2_q)
        p3_q = p3_q / torch.linalg.norm(p3_q)

        w3_x_x, w3_x_y, w3_x_z = 10, 5, 80 #experiment2
        # w1_x_x, w1_x_y, w1_x_z, w2_x_x, w2_x_y, w2_x_z, w3_x_x, w3_x_y, w3_x_z = 3, 3, 20, 3, 3, 20, 10, 10, 100
        # w1_q_w, w1_q_p, w1_q_q, w1_q_r, w2_q_w, w2_q_p, w2_q_q, w2_q_r, w3_q_w, w3_q_p, w3_q_q, w3_q_r = 150, 150, 150, 

        pose_x_x, pose_x_y, pose_x_z = pose_x[:, 0], pose_x[:, 1], pose_x[:, 2]
        p3_x_x, p3_x_y, p3_x_z = p3_x[:, 0], p3_x[:, 1], p3_x[:, 2]


        l3_x_x = self.ed(pose_x_x, p3_x_x) * w3_x_x
        l3_x_y = self.ed(pose_x_y, p3_x_y) * w3_x_y
        l3_x_z = self.ed(pose_x_z, p3_x_z) * w3_x_z

        l1_q = self.ed(pose_q, p1_q) * self.w1_q
        l2_q = self.ed(pose_q, p2_q) * self.w2_q
        l3_q = self.ed(pose_q, p3_q) * self.w3_q

        loss = l3_x_x + l3_x_y + l3_x_z + l1_q + l2_q + l3_q

        return loss

def posenet_v1():
    model = PoseNet()
    return model
