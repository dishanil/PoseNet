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
import time

learning_rate = 0.0001
train_batch_size = 1
test_batch_size = 1
EPOCH = 80000
directory = '/home/dishani/code/posenet/SmithHall/1604/1604/'
# directory = '/home/dishani/code/posenet/baseline/pytorch-posenet-master/DataSet/KingsCollege/'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
writer = SummaryWriter('/home/dishani/code/posenet/baseline/pytorch-posenet-master/Experiments/inceptionnet_smith_world_timestampus/runs')

ckpt_path = '/home/dishani/code/posenet/baseline/pytorch-posenet-master/Experiments/inceptionnet_smith_world_timestampus/ckpts/'

datasource = DataSource(directory, train=True)
train_loader = Data.DataLoader(dataset=datasource, batch_size=train_batch_size, shuffle=False)
print("len(train_loader) = ", len(train_loader))


test_datasource = DataSource(directory, train=False)
test_loader = Data.DataLoader(dataset=test_datasource, batch_size=test_batch_size, shuffle=False)
print("len(test_loader) = ", len(test_loader))

train_output_file = '/home/dishani/code/posenet/SmithHall/1604/1604/train_output_world_timestampus_scale1_2_2.txt'

test_output_file = '/home/dishani/code/posenet/SmithHall/1604/1604/test_output_world_timestampus_scale100_2.txt'

def inverse_quaternion(q):
    """
    Computes the inverse of a quaternion q.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # w, x, y, z = q

    norm_squared = w*w + x*x + y*y + z*z
    return np.transpose(np.array([w, -x, -y, -z]) / norm_squared)

def quaternion_rotate(q, q_rot):
    """
    Applies a rotation to a quaternion q using another quaternion q_rot.
    """
    q = np.array(q.detach().cpu(), dtype=np.float64)
    q_rot = np.array(q_rot.detach().cpu(), dtype=np.float64)

    q_new = quaternion_multiply(quaternion_multiply(q_rot, q), inverse_quaternion(q_rot))
    return torch.tensor(q_new).to(device)

def quaternion_multiply(q1, q2):
    """
    Computes the product of two quaternions q1 and q2.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    # w1, x1, y1, z1 = q1
    # w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2

    return np.transpose(np.array([w, x, y, z]))
    # return np.array([w, x, y, z])

def apply_inverse_rotation(q, r):
    """
    Apply inverse of the rotation in quaternion `r` on quaternion `q`
    """
    q = np.array(q.detach().cpu(), dtype=np.float64)
    q_norm = np.linalg.norm(q, 1)
    # q_norm = np.linalg.norm(q)
    q = q / q_norm

    r = np.array(r.detach().cpu(), dtype=np.float64) 
    r_norm = np.linalg.norm(r, 1)
    # r_norm = np.linalg.norm(r)
    r = r / r_norm
    # r_conj = np.transpose(np.array([r[:, 0], -r[:, 1], -r[:, 2], -r[:, 3]]))
    
    r_inv = inverse_quaternion(r)

    q_new = quaternion_multiply(quaternion_multiply(r_inv, q), r)
    # q_new = r_inv * q * r
    return torch.tensor(q_new).to(device)

def investigate(test=True):
    loader = None

    if(test):
        loader = test_loader
    else:
        loader = train_loader

    results = torch.zeros(len(loader) * train_batch_size, 4)
    
    j = 0

    for step, (images, poses) in enumerate(loader):
        
        for i in range(7):
            poses[i] = np.array(poses[i])
        poses = np.transpose(poses)
        b_poses = Variable(torch.Tensor(poses), requires_grad=False).to(device)

        pose_q = b_poses[:, 0:4].to(device=device)

        results[j : j + pose_q.shape[0]] = pose_q

        j = j + pose_q.shape[0]

        if(step%50==0):
            print('Step:  ', step, '  Running mean :  ', torch.mean(results, axis=0), ' and var = ', torch.var(results, axis=0))

    mean_result = torch.mean(results, axis=0)
    print('mean :  ', torch.mean(results, axis=0))
    print('var :  ', torch.var(results, axis=0))
    
from scipy.spatial.transform import Rotation

def euler_angle_diff(q1, q2, degrees=True):
    """
    Compute the Euler angle difference between two batches of quaternions `q1` and `q2`.
    """
    q1 = q1.cpu().numpy()
    q2 = q2.detach().cpu().numpy()

    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    diff = r1.inv() * r2
    angles = diff.as_euler('xyz')
    if degrees:
        angles = np.rad2deg(angles)
    return angles

def test(model, epoch, test=True):
    model.eval()

    loader = None
    
    if(test):
        loader = test_loader
        output_file = test_output_file
    else:
        loader = train_loader
        output_file = train_output_file

    results = torch.zeros(len(loader), 3)
        
    for step, (img_path, images, poses) in enumerate(loader):
        b_images = Variable(images, requires_grad=False).to(device)
        for i in range(7):
            poses[i] = np.array(poses[i])
        poses = np.transpose(poses)
        b_poses = Variable(torch.Tensor(poses), requires_grad=False).to(device)

        # pose_x = b_poses[:, 0:3]
        # pose_q = b_poses[:, 3:]

        scaling_factor = 100

        pose_x = b_poses[:, 4:7].to(device=device)
        pose_q = b_poses[:, 0:4].to(device=device) / scaling_factor

        start_time = time.time()
        _,_,_,_, predicted_x, predicted_q = model(b_images)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Model took {duration:.2f} seconds to run.")

        predicted_q = predicted_q.to(device=device) / scaling_factor
        predicted_x = predicted_x.to(device=device)

        timestamp = img_path[0].split('/')[-1][6:-4]
        # print(timestamp)

        # timestamp = txt_path[0].split('/')[-1].split('_')[-1].split('.')[0]

        # lines = [timestamp, ' ', str(predicted_q[0].item()), ' ', str(predicted_q[1].item()), ' ', str(predicted_q[2].item()), ' ', str(predicted_q[3].item()), ' ',  str(predicted_x[0].item()), ' ', str(predicted_x[1].item()), ' ', str(predicted_x[2].item()), '\n']

        # with open(output_file, "a") as f:
        #     f.writelines(lines)

        # pose_q_world = b_poses[:, 7:11].to(device=device)
        # mat_q_world_device = b_poses[:, 11:15].to(device=device)

        # quat_extrinsics = quaternion_rotate(pose_q_world, mat_q_world_device)

        # quat_extrinsics = quaternion_rotate(mat_q_world_device[0], pose_q_world[0])

        # predicted_q_world = apply_inverse_rotation(predicted_q, mat_q_world_device)

        # q1_world = pose_q_world / torch.linalg.norm(pose_q_world)
        # q2_world = predicted_q_world / torch.linalg.norm(predicted_q_world)
        # d_world = abs(torch.sum(torch.multiply(q1_world,q2_world)))
        # theta_world = 2 * torch.arccos(d_world) * 180/math.pi

        # q1 = pose_q / torch.linalg.norm(pose_q)
        # q2 = predicted_q / torch.linalg.norm(predicted_q)
        # d = abs(torch.sum(torch.multiply(q1,q2)))
        # theta = 2 * torch.arccos(d) * 180/math.pi

        # theta = torch.tensor(euler_angle_diff(q1, q2)).to(device)

        # theta = torch.mean(torch.linalg.norm(theta, dim=1), axis=0)

        # error_x = torch.mean(torch.linalg.norm(pose_x - predicted_x, dim=1), axis=0)
        # error_q = torch.mean(torch.linalg.norm(pose_q - predicted_q, dim=1), axis=0)

        # results[step, 0] = error_x.item()
        # results[step, 1] = theta.item()
        # results[step, 2] = error_q.item()
        # results[step, 2] = theta_world.item()

        # if(step%50==0):
        #     # print("Differnce between DEVICE poses", torch.linalg.norm(quat_extrinsics - pose_q))

        #     print('Step:  ', step, '  Running mean Error XYZ (m):  ', torch.mean(results[:, 0]), '  Error Q (degrees):  ', torch.mean(results[:, 1]), '  Error Q (DIFF):  ', torch.mean(results[:, 2]))

    # mean_result = torch.mean(results, axis=0)
    # print('TESTING === Epoch: ', epoch, 'Mean error: ', torch.mean(results[:, 0]), 'm  and ', torch.mean(results[:, 1]), 'degrees in world coords.', '  Error Q (DIFF):  ', torch.mean(results[:, 2]))
    # print("Diff in world coords = ", torch.mean(results[:, 2]), " degrees.")

    # writer.add_scalar("Loss/test_error_x", torch.mean(results[:, 0]), epoch)
    # writer.add_scalar("Loss/test_error_angle", torch.mean(results[:, 1]), epoch)
    # writer.add_scalar("Loss/test_error_angle_world", torch.mean(results[:, 2]), epoch)

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

class LePoseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.og_model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

        self.model = torch.nn.Sequential(*(list(self.og_model.children())[:-1])).to(device=device)

        self.aux1_xyz = InceptionAux(512, 3, dropout=0.7).to(device=device)
        self.aux1_wpqr = InceptionAux(512, 4, dropout=0.7).to(device=device)

        self.aux2_xyz = InceptionAux(528, 3, dropout=0.7).to(device=device)
        self.aux2_wpqr = InceptionAux(528, 4, dropout=0.7).to(device=device)

        self.cls_fc_pose_xyz = nn.Linear(1024, 3).to(device=device)
        self.cls_fc_pose_wpqr = nn.Linear(1024, 4).to(device=device)

    def forward(self, x):
        # TODO return unnormalized log-probabilities here
        self.og_model.inception4a.register_forward_hook(get_activation('inception4a'))
        self.og_model.inception4d.register_forward_hook(get_activation('inception4d'))

        pre_output = self.model(x.to(device=device)).squeeze()

        cls_xyz = self.cls_fc_pose_xyz(pre_output)
        cls_wpqr = self.cls_fc_pose_wpqr(pre_output)

        output_4a = activation['inception4a']

        cls_inter_1_xyz = self.aux1_xyz(output_4a)
        cls_inter_1_wpqr = self.aux1_wpqr(output_4a)

        output_4d = activation['inception4d']

        cls_inter_2_xyz = self.aux2_xyz(output_4d)
        cls_inter_2_wpqr = self.aux2_wpqr(output_4d)

        return cls_inter_1_xyz, cls_inter_1_wpqr, cls_inter_2_xyz, cls_inter_2_wpqr, cls_xyz, cls_wpqr

def main(train=True, PATH=''):
    # train dataset and train loader

    if(not train):
        print("Testing Start")
        model = LePoseNet()
        
        for epoch in np.arange(10, 11, 10):
            print("Epoch = ", epoch)
            ckpt_path = PATH + str(epoch) + '.pt'
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:1')), strict=False)
            model.eval()
            # test(model, epoch, False)
            test(model, epoch, True)
            
        print("Testing done")
        return

    # posenet
    
    posenet = LePoseNet().to(device=device) 

    for param in posenet.parameters():
        param.requires_grad = True

    posenet.train()

    # loss function
    criterion = PoseNet.PoseLoss(3, 3, 1, 250, 250, 600)
    # criterion = PoseNet.PoseLoss(3, 3, 1, 400, 400, 1000)

    # load pre-trained model

    # train the network
    optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(EPOCH):
        for step, (images, poses) in enumerate(train_loader):
            posenet.train()

            b_images = Variable(images, requires_grad=True).to(device)

            for i in range(7):
                poses[i] = np.array(poses[i])

            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print("iteration: " + str(epoch) + "\n    " + "Loss is: " + str(loss))
                writer.add_scalar("Loss/train", loss, epoch)
                test(posenet, epoch, True)

        if epoch%2 == 0 :
            ckpt_path = '/home/dishani/code/posenet/baseline/pytorch-posenet-master/Experiments/inceptionnet_smith_world_timestampus/ckpts/'
            torch.save(posenet.state_dict(), ckpt_path + str(epoch) + '.pt')

        test(posenet, epoch, True)

        # if epoch%10 == 0 :
        #     scheduler.step()
        #     print("LR changed at epoch: " + str(epoch) + "\n    " + "LR is: " + str(scheduler.get_last_lr()[0]))
        #     writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch)
            
        

if __name__ == '__main__':
    # investigate()
    train = False
    main(train, '/home/dishani/code/posenet/baseline/pytorch-posenet-master/Experiments/inceptionnet_smith_world_timestampus/ckpts/')
