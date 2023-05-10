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
import wideresnet

learning_rate = 0.000001
batch_size = 20
EPOCH = 80000

# directory = '/home/dishani/code/posenet/SmithHall/1604/1604/'
directory = '/home/dishani/code/posenet/baseline/pytorch-posenet-master/DataSet/KingsCollege/'
# train dataset and train loader
datasource = DataSource(directory, train=True)
train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

test_datasource = DataSource(directory, train=False)
test_loader = Data.DataLoader(dataset=test_datasource, batch_size=batch_size, shuffle=True)

ckpt_path = '/home/dishani/code/posenet/baseline/pytorch-posenet-master/Experiments/resnet18_kings/ckpts/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter('/home/dishani/code/posenet/baseline/pytorch-posenet-master/Experiments/resnet18_kings/runs/')

def test_train(model, epoch):
    model.eval()

    results = torch.zeros(len(train_loader), 2)
    results_x = torch.zeros(len(train_loader), 3)

    var_x_test = 0
    var_x_train = 0
    mean_x_test = 0
    mean_x_train = 0

    var_y_test = 0
    var_y_train = 0

    var_z_test = 0
    var_z_train = 0
    print("len(train_loader) = ", len(train_loader))

    for step, (images, poses) in enumerate(train_loader):
        # print("step = ", step)
        b_images = Variable(images, requires_grad=False).to(device)
        # print("b_images.shape = ", b_images.shape)
        poses[0] = np.array(poses[0])
        poses[1] = np.array(poses[1])
        poses[2] = np.array(poses[2])
        poses[3] = np.array(poses[3])
        poses[4] = np.array(poses[4])
        poses[5] = np.array(poses[5])
        poses[6] = np.array(poses[6])
        poses = np.transpose(poses)
        b_poses = Variable(torch.Tensor(poses), requires_grad=False).to(device)

        # pose_x = b_poses[:, 0:3]
        # pose_q = b_poses[:, 3:]

        pose_x = b_poses[:, 4:].to(device=device)
        pose_q = b_poses[:, 0:4].to(device=device)

        predicted_x, predicted_q = model(b_images)
        predicted_q = predicted_q.to(device=device)
        predicted_x = predicted_x.to(device=device)

        # lines = [img_path[0].split('/')[-1][6:-4], ' ', str(predicted_q[0].item()), ' ', str(predicted_q[1].item()), ' ', str(predicted_q[2].item()), ' ', str(predicted_q[3].item()), ' ',  str(predicted_x[0].item()), ' ', str(predicted_x[1].item()), ' ', str(predicted_x[2].item()), '\n']

        # with open(train_output_file, "a") as f:
        #     f.writelines(lines)

        # q1 = F.normalize(pose_q, p=2, dim=1)
        # q1 = pose_q / torch.linalg.norm(pose_q, dim=1).reshape(-1,1).repeat(1,4)
        q1 = pose_q / torch.linalg.norm(pose_q)
        # q2 = F.normalize(predicted_q, p=2, dim=1)
        # q2 = predicted_q / torch.linalg.norm(predicted_q, dim=1).reshape(-1,1).repeat(1,4)
        q2 = predicted_q / torch.linalg.norm(predicted_q)
        d = abs(torch.sum(torch.multiply(q1,q2)))
        theta = 2 * torch.arccos(d) * 180/math.pi

        error_x = torch.mean(torch.linalg.norm(pose_x - predicted_x, dim=1), axis=0)

        # error_x_x = torch.mean(torch.abs(pose_x[:,0] - predicted_x[:,0]))
        # error_x_y = torch.mean(torch.abs(pose_x[:,1] - predicted_x[:,1]))
        # error_x_z = torch.mean(torch.abs(pose_x[:,2] - predicted_x[:,2]))

        # var_x_test += torch.var(predicted_x[:,0])
        # var_x_train += torch.var(pose_x[:,0])

        # var_y_test += torch.var(predicted_x[:,1])
        # var_y_train += torch.var(pose_x[:,1])

        # var_z_test += torch.var(predicted_x[:,2])
        # var_z_train += torch.var(pose_x[:,2])

        # mean_x_test += torch.mean(predicted_x[:,0])
        # mean_x_train += torch.mean(pose_x[:,0])

        # results_x[step, 0] = error_x_x.item()
        # results_x[step, 1] = error_x_y.item()
        # results_x[step, 2] = error_x_z.item()

        results[step, 0] = error_x.item()
        results[step, 1] = theta.item()

        print('Step:  ', step, '  Running mean Error XYZ (m):  ', torch.mean(results[:, 0]), '  Error Q (degrees):  ', torch.mean(results[:, 1]))

    # print("Outside")
    mean_result = torch.mean(results, axis=0)
    print('TESTING === Epoch: ', epoch, 'Mean error: ', mean_result[0], 'm  and ', mean_result[1], 'degrees.')

    # mean_result_x = torch.mean(results_x, axis=0)
    # print('TESTING === Epoch: ', epoch, 'Mean error X: ', mean_result_x[0], 'm ; Mean error Y: ', mean_result_x[1], 'Mean error Z: ', mean_result_x[2])
    
    # print('Variance of X PREDICTED: ', var_x_test/len(test_loader), 'Variance of X GT: ', var_x_train/len(test_loader), 'Mean of X PREDICTED: ', mean_x_test/len(test_loader), 'Mean of X GT: ', mean_x_train/len(test_loader))

    # print('Variance of Y PREDICTED: ', var_y_test/len(test_loader), 'Variance of Y GT: ', var_y_train/len(test_loader))

    # print('Variance of Z PREDICTED: ', var_z_test/len(test_loader), 'Variance of Y GT: ', var_z_train/len(test_loader))

    # writer.add_scalar("Loss/test_error_x_x", mean_result_x[0], epoch)
    # writer.add_scalar("Loss/test_error_x_y", mean_result_x[1], epoch)
    # writer.add_scalar("Loss/test_error_x_z", mean_result_x[2], epoch)

    # writer.add_scalar("Loss/test_error_x", mean_result[0], epoch)
    # writer.add_scalar("Loss/test_error_angle", mean_result[1], epoch)

def test(model, epoch):
    model.eval()

    results = torch.zeros(len(test_loader), 2)
    results_x = torch.zeros(len(test_loader), 3)

    var_x_test = 0
    var_x_train = 0
    mean_x_test = 0
    mean_x_train = 0

    var_y_test = 0
    var_y_train = 0

    var_z_test = 0
    var_z_train = 0
    # print("len(test_loader) = ", len(test_loader))

    for step, (images, poses) in enumerate(test_loader):
        # print("step = ", step)
        b_images = Variable(images, requires_grad=False).to(device)
        # print("b_images.shape = ", b_images.shape)
        poses[0] = np.array(poses[0])
        poses[1] = np.array(poses[1])
        poses[2] = np.array(poses[2])
        poses[3] = np.array(poses[3])
        poses[4] = np.array(poses[4])
        poses[5] = np.array(poses[5])
        poses[6] = np.array(poses[6])
        poses = np.transpose(poses)
        b_poses = Variable(torch.Tensor(poses), requires_grad=False).to(device)

        pose_x = b_poses[:, 0:3].to(device=device)
        pose_q = b_poses[:, 3:].to(device=device)

        # pose_x = b_poses[:, 4:].to(device=device)
        # pose_q = b_poses[:, 0:4].to(device=device)

        predicted_x, predicted_q = model(b_images)
        predicted_q = predicted_q.to(device=device)
        predicted_x = predicted_x.to(device=device)

        # lines = [img_path[0].split('/')[-1][6:-4], ' ', str(predicted_q[0].item()), ' ', str(predicted_q[1].item()), ' ', str(predicted_q[2].item()), ' ', str(predicted_q[3].item()), ' ',  str(predicted_x[0].item()), ' ', str(predicted_x[1].item()), ' ', str(predicted_x[2].item()), '\n']

        # with open(test_output_file, "a") as f:
        #     f.writelines(lines)

        # q1 = F.normalize(pose_q, p=2, dim=1)
        # q1 = pose_q / torch.linalg.norm(pose_q, dim=1).reshape(-1,1).repeat(1,4)
        q1 = pose_q / torch.linalg.norm(pose_q)
        # q2 = F.normalize(predicted_q, p=2, dim=1)
        # q2 = predicted_q / torch.linalg.norm(predicted_q, dim=1).reshape(-1,1).repeat(1,4)
        q2 = predicted_q / torch.linalg.norm(predicted_q)
        d = abs(torch.sum(torch.multiply(q1,q2)))
        theta = 2 * torch.arccos(d) * 180/math.pi

        error_x = torch.mean(torch.linalg.norm(pose_x - predicted_x, dim=1), axis=0)

        # error_x_x = torch.mean(torch.abs(pose_x[:,0] - predicted_x[:,0]))
        # error_x_y = torch.mean(torch.abs(pose_x[:,1] - predicted_x[:,1]))
        # error_x_z = torch.mean(torch.abs(pose_x[:,2] - predicted_x[:,2]))

        # var_x_test += torch.var(predicted_x[:,0])
        # var_x_train += torch.var(pose_x[:,0])

        # var_y_test += torch.var(predicted_x[:,1])
        # var_y_train += torch.var(pose_x[:,1])

        # var_z_test += torch.var(predicted_x[:,2])
        # var_z_train += torch.var(pose_x[:,2])

        # mean_x_test += torch.mean(predicted_x[:,0])
        # mean_x_train += torch.mean(pose_x[:,0])

        # results_x[step, 0] = error_x_x.item()
        # results_x[step, 1] = error_x_y.item()
        # results_x[step, 2] = error_x_z.item()

        results[step, 0] = error_x.item()
        results[step, 1] = theta.item()

        print('Step:  ', step, '  Running mean Error XYZ (m):  ', torch.mean(results[:, 0]), '  Error Q (degrees):  ', torch.mean(results[:, 1]))

    # print("Outside")
    mean_result = torch.mean(results, axis=0)
    print('TESTING === Epoch: ', epoch, 'Mean error: ', mean_result[0], 'm  and ', mean_result[1], 'degrees.')

    # mean_result_x = torch.mean(results_x, axis=0)
    # print('TESTING === Epoch: ', epoch, 'Mean error X: ', mean_result_x[0], 'm ; Mean error Y: ', mean_result_x[1], 'Mean error Z: ', mean_result_x[2])
    
    # print('Variance of X PREDICTED: ', var_x_test/len(test_loader), 'Variance of X GT: ', var_x_train/len(test_loader), 'Mean of X PREDICTED: ', mean_x_test/len(test_loader), 'Mean of X GT: ', mean_x_train/len(test_loader))

    # print('Variance of Y PREDICTED: ', var_y_test/len(test_loader), 'Variance of Y GT: ', var_y_train/len(test_loader))

    # print('Variance of Z PREDICTED: ', var_z_test/len(test_loader), 'Variance of Y GT: ', var_z_train/len(test_loader))

    # writer.add_scalar("Loss/test_error_x_x", mean_result_x[0], epoch)
    # writer.add_scalar("Loss/test_error_x_y", mean_result_x[1], epoch)
    # writer.add_scalar("Loss/test_error_x_z", mean_result_x[2], epoch)

    writer.add_scalar("Loss/test_error_x", mean_result[0], epoch)
    writer.add_scalar("Loss/test_error_angle", mean_result[1], epoch)
    
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

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module.cuda()

class ResNet_pretrained(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.model_file = 'wideresnet18_places365.pth.tar'

        # self.model = wideresnet.resnet18(num_classes=365)
        # checkpoint = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # self.model.load_state_dict(state_dict)

        self.model = wideresnet.resnet18()

        for i, (name, module) in enumerate(self.model._modules.items()):
            module = recursion_change_bn(self.model)

        self.model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0).cuda()

        self.cls_fc_pose_xyz = InceptionAux(512, 3).cuda()
        self.cls_fc_pose_wpqr = InceptionAux(512, 4).cuda()

    def forward(self, x):

        # self.model.layer4.register_forward_hook(get_activation('layer4'))
        self.model.avgpool.register_forward_hook(get_activation('avgpool'))

        pre_output = self.model(x.cuda())

        pre_output = activation['avgpool']

        cls_xyz = self.cls_fc_pose_xyz(pre_output)
        cls_wpqr = self.cls_fc_pose_wpqr(pre_output)

        return cls_xyz, cls_wpqr



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_x_1 = nn.Linear(512*ResBlock.expansion, 128)
        self.fc_q_1 = nn.Linear(512*ResBlock.expansion, 128)

        self.fc_x = nn.Linear(128, 3)
        self.fc_q = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x_x_1 = self.fc_x_1(x)
        x_q_1 = self.fc_q_1(x)

        pose_x = self.fc_x(x_x_1)
        pose_q = self.fc_q(x_q_1)
        
        return pose_x, pose_q
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
def ResNet50(channels=3):
    return ResNet(Bottleneck, [3,4,6,3], channels)

class PoseLoss_resnet(nn.Module):

    def __init__(self, w3_x, w3_q):
        super(PoseLoss_resnet, self).__init__()

        self.w3_x = w3_x
        self.w3_q = w3_q
        self.ed = nn.MSELoss()

    def forward(self, p3_x, p3_q, poseGT):
        pose_x = poseGT[:, 0:3]
        pose_q = poseGT[:, 3:]

        # pose_q = poseGT[:, 0:4]
        # pose_x = poseGT[:, 4:]

        pose_q = pose_q / torch.linalg.norm(pose_q)
        p3_q = p3_q / torch.linalg.norm(p3_q)

        pose_x_x, pose_x_y, pose_x_z = pose_x[:, 0], pose_x[:, 1], pose_x[:, 2]
        p3_x_x, p3_x_y, p3_x_z = p3_x[:, 0], p3_x[:, 1], p3_x[:, 2]

        l3_x_x = self.ed(pose_x_x, p3_x_x) * self.w3_x * 3
        l3_x_y = self.ed(pose_x_y, p3_x_y) * self.w3_x * 2
        l3_x_z = self.ed(pose_x_z, p3_x_z) * self.w3_x

        l3_q = self.ed(pose_q, p3_q) * self.w3_q 

        loss = l3_x_x + l3_x_y + l3_x_z + l3_q

        return loss

def main(train=True, PATH=''):
    # posenet
    if(not train):
        print("Testing Start")
        model = ResNet_pretrained()
        
        for epoch in np.arange(5450, 5451, 10):
            print("Epoch = ", epoch)
            ckpt_path = PATH + str(epoch) + '.pt'
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')), strict=False)
            model.eval()
            test(model, epoch)
            test_train(model, epoch)
        print("Testing done")
        return

    posenet = ResNet_pretrained().cuda()

    for param in posenet.parameters():
        param.requires_grad = True

    posenet.train()

    # loss function
    criterion = PoseLoss_resnet(1, 1000)

    # load pre-trained model

    # train the network
    # optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()), lr=learning_rate)
    optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()), lr=learning_rate, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(EPOCH):
        for step, (_, images, poses) in enumerate(train_loader):
            posenet.train()

            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            p3_x, p3_q = posenet(b_images)
            loss = criterion(p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 420 == 0:
                print("iteration: " + str(epoch) + "\n    " + "Loss is: " + str(loss))
        
        writer.add_scalar("Loss/train", loss, epoch)

        test(posenet, epoch)
        # posenet.train()

        # if (epoch+1)%5 == 0 :
        #     scheduler.step()
        #     print("LR changed at epoch: " + str(epoch) + "\n    " + "LR is: " + str(scheduler.get_last_lr()[0]))
        #     writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch)

        if epoch%10 == 0 :
            torch.save(posenet.state_dict(), ckpt_path + str(epoch) + '.pt')


if __name__ == '__main__':
    main(False, ckpt_path)
