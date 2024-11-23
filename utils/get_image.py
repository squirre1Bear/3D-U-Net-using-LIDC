import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import cv2
import pylidc as pl
from pylidc.utils import consensus
import os
import torch

# 输入文件夹路径，得到读取后的图像、掩膜张量。
def get_tensor(dataset_path):
    ii = 0
    # 这个是1012个切片的最大尺寸，用于将不同切片填充成同样大小。
    # 大小都是8的倍数，是为了最大池化的时候尺寸大小不会出现小数，防止特征图大小不同无法合并。
    max_d = 72
    max_h = 88
    max_w = 104
    image_list = []
    cmask_list = []
    wrong_dicom = []
    # Query for a scan, and convert it to an array volume.
    for dicom_name in os.listdir(dataset_path):
        print(dicom_name)
        PathDicom = os.path.join(dicom_name)    # 获得每个Dicom文件的路径
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == dicom_name).first()    # 按名字读取Dicom文件
        vol = scan.to_volume()   # 转换成3D数组,512*512*Z，Z的大小不定

        # Cluster the annotations for the scan, and grab one.
        nods = scan.cluster_annotations()    # 获取聚类后的annotation
        try:    # 防止有的文件没有annotation
            anns = nods[0]
            Malignancy = anns[0]
            Malignancy = Malignancy.Malignancy
        except:
            pass
        # Perform a consensus consolidation and 50% agreement level.
        # We pad the slices to add context for viewing.
        cmask, cbbox, masks = consensus(anns, clevel=0.5,
                                        pad=[(0, 0), (7, 7), (20, 20)])

        # 下面的image是原始图像，数组！cmask是对应的掩码(true/false)！
        image = vol[cbbox]   # 用cbbox定义的边界框，从体积数据vol中提取切片。是52*58*49的数组
        image = normalize_hu(image)
        # 开始尺寸填充
        pad = ((0, max_d - image.shape[0]), (0, max_h - image.shape[1]), (0, max_w - image.shape[2]))
        image = np.pad(image, pad, mode='constant', constant_values=0)
        cmask = np.pad(cmask, pad, mode='constant', constant_values=0)

        if image.shape != (72, 88, 104) or cmask.shape != (72, 88, 104):
            print("有个形状错误的数据"+dicom_name)
            wrong_dicom.append(dicom_name)
            wrong_dicom.append(image.shape)
            wrong_dicom.append(cmask.shape)
            continue

        device = torch.device('cuda:0')
        # img_tensor = torch.from_numpy(image).float().to(device)  # 确保数据类型
        # labels_tensor = torch.from_numpy(cmask).long().to(device)

        image_list.append(image)
        cmask_list.append(cmask)
    # 不同切片的image大小不同，需要找出最大尺寸，padding成一样大小的。
        print(image.shape)
        print(cmask.shape)
        ii += 1
    # 返回是list，里面存的数据是相同大小（72*85*97）的图片image
    return image_list, cmask_list, wrong_dicom

# 归一化
def normalize_hu(image):     # 图像是int16类型，总共49个通道，和z的范围大小一样。每张图高52，宽58
    # 将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

# image_list, cmask_list,wrong_dicom = get_tensor(dataset_path=r"E:\LIDC_dataset\train_dataset")
# np.save(r'E:\LIDC_dataset\train_image_list.npy', image_list)
# np.save(r'E:\LIDC_dataset\train_label_list.npy', cmask_list)
# print("数据大小有误被剔除的文件：")
# for dicom_name in wrong_dicom:
#     print(dicom_name)

image_list, cmask_list, wrong_dicom = get_tensor(dataset_path=r"E:\LIDC_dataset\test_dataset")
np.save(r'E:\LIDC_dataset\test_image_list.npy', image_list)
np.save(r'E:\LIDC_dataset\test_label_list.npy', cmask_list)

print("数据大小有误被剔除的文件：")
for dicom_name in wrong_dicom:
    print(dicom_name)
# data = np.load(r'E:\LIDC_dataset\test_image_list.npy')可以读取list