# import os
# import os.path
# import torch.utils.data as data
# from torchvision import transforms as T
# from PIL import Image
# import numpy as np

# class DataSource(data.Dataset):
#     def __init__(self, root, train=True, transforms=None):
#         self.root = os.path.expanduser(root)
#         self.transforms = transforms
#         self.train = train

#         self.image_poses = []
#         self.images_path = []
#         self.txts_path = []

#         self._get_data()

#         if transforms is None:
#             normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#             if not train:
#                 self.transforms = T.Compose(
#                     [T.Resize((224, 224)),
#                      T.ToTensor(),
#                      normalize]
#                 )
#             else:
#                 self.transforms = T.Compose(
#                     [T.Resize((224, 224)),
#                      T.ToTensor(),
#                      normalize]
#                 )

#     def _get_data(self):

#         if self.train:
#             txt_file = self.root + 'train_dataset_world_timestampus.txt'
#         else:
#             txt_file = self.root + 'test_dataset_world_timestampus.txt'

#         # txt_file = self.root + 'dataset_combined_world.txt'

#         print("Using data file = ", txt_file)

#         with open(txt_file, 'r') as f:
#             next(f)  # skip the 3 header lines
#             next(f)
#             next(f)
#             # for _ in range(8):
#             #     next(f)

#             for curr_line in f:
#                 # rotation = np.zeros((3,3), dtype=np.float32)
#                 quaternion = np.zeros((4), dtype=np.float32)
#                 translation = np.zeros((3), dtype=np.float32)

#                 txtname, fname, quaternion[0], quaternion[1], quaternion[2], quaternion[3], translation[0], translation[1], translation[2]  = curr_line.split(' ')
                
            
#                 self.image_poses.append((quaternion[0], quaternion[1], quaternion[2], quaternion[3], translation[0], translation[1], translation[2]))

#                 self.images_path.append(fname)

#                 self.txts_path.append(txtname)

#         self.image_poses = [x for _, _, x in sorted(zip(self.images_path, self.txts_path, self.image_poses))]

#         self.images_path = [x for x, _, _ in sorted(zip(self.images_path, self.txts_path, self.image_poses))]

#         self.txts_path = [x for _, x, _ in sorted(zip(self.images_path, self.txts_path, self.image_poses))]

#         # print(len(self.images_path))

#     def __getitem__(self, index):
#         """
#         return the data of one image
#         """
#         txt_path = os.path.join(self.root, self.txts_path[index])
#         img_path = os.path.join(self.root, self.images_path[index])
#         img_pose = self.image_poses[index]
#         data = Image.open(img_path)
#         data = self.transforms(data)
#         return txt_path, img_path, data, img_pose

#     def __len__(self):
#         return len(self.images_path)


import os
import os.path
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image


class DataSource(data.Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if not train:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.CenterCrop(256),
                     T.ToTensor(),
                     normalize]
                )
            else:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.RandomCrop(256),
                     T.ToTensor(),
                     normalize]
                )

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'train_dataset_world_timestampus.txt'
        else:
            txt_file = self.root + 'test_dataset_world_timestampus.txt'

        # if self.train:
        #     txt_file = self.root + 'dataset_train.txt'
        # else:
        #     txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            # next(f)  # skip the 3 header lines
            # next(f)
            # next(f)
            for line in f:
                # _, fname, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = line.split()
                # p0 = float(p0)
                # p1 = float(p1)
                # p2 = float(p2)
                # p3 = float(p3)
                # p4 = float(p4)
                # p5 = float(p5)
                # p6 = float(p6)
                # p7 = float(p7)
                # p8 = float(p8)
                # p9 = float(p9)
                # p10 = float(p10)
                # p11 = float(p11)
                # p12 = float(p12)
                # p13 = float(p13)
                # p14 = float(p14)

                # self.image_poses.append((p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14))

                scaling_factor = 100

                _, fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0) * scaling_factor
                p1 = float(p1) * scaling_factor
                p2 = float(p2) * scaling_factor
                p3 = float(p3) * scaling_factor
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)

                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))

                # self.images_path.append(self.root + fname)
                self.images_path.append(fname)

    def __getitem__(self, index):
        """        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return img_path, data, img_pose

    def __len__(self):
        return len(self.images_path)