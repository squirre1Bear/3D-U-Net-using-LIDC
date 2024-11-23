import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import cv2
import pylidc as pl
from pylidc.utils import consensus
import os

dataset_path = r"C:\Users\91604\Desktop\lungSeg\LIDC\manifest-1730534256759\LIDC-IDRI"
dicom_name = 'LIDC-IDRI-0001'

# 归一化
def normalize_hu(image):     # 图像是int16类型，总共49个通道，和z的范围大小一样。每张图高52，宽58
    # 将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

ii = 0
# Query for a scan, and convert it to an array volume.
for dicom_name in os.listdir(dataset_path):
    print(dicom_name)
    PathDicom = os.path.join(dicom_name)
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == dicom_name).first()
    vol = scan.to_volume()
    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    try:
        anns = nods[0]
        Malignancy = anns[0]
        Malignancy = Malignancy.Malignancy
    except:
        pass
    # Perform a consensus consolidation and 50% agreement level.
    # We pad the slices to add context for viewing.
    cmask, cbbox, masks = consensus(anns, clevel=0.5,
                                    pad=[(0, 0), (7, 7), (20, 20)])
    image = vol[cbbox]   # 用cbbox定义的边界框，从体积数据vol中提取切片
    '''
    下面这段可以显示原始图像
    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))  # 选中z方向上索引的中间值，用于选择要可视化的切片

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=1)
    plt.tight_layout()  # 调整布局以适应图像
    plt.show()  # 显示图像
    cc = cmask[:,:,25]   # 这句是显示掩膜信息的，25是z方向的中值
    qq = image[:,:,25]   # 显示图像的灰度信息，范围是【0，1】
    exit()
    '''
    image = normalize_hu(image)    # 大小为52*58*49，类型是nparray

    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))  # 选中z方向上索引的中间值，用于选择要可视化的切片
    if Malignancy == 'Highly Unlikely':
        label1 = 1
    elif Malignancy == 'Moderately Unlikely':
        label1 = 2
    elif Malignancy == 'Indeterminate':
        label1 = 3
    elif Malignancy == 'Moderately Suspicious':
        label1 = 4
    elif Malignancy == 'Highly Suspicious':
        label1 = 5
    print(label1)

    # 矩阵增广和传参
    ArrayDicom = image
    ArrayDicom_mask = cmask

    # 中心点定位
    x_, y_, z_ = np.shape(image)
    x_ = int(x_ / 2)
    y_ = int(y_ / 2)
    z_ = int(z_ / 2)

    # 图像进行切片处
    save_dir = r"C:\Users\91604\Desktop\lungSeg\LIDC\train_anno"  # 修改为标签存放的位置

    # 检查并创建文件夹
    train_dirs = ['z', 'x', 'y']
    for directory in train_dirs:
        full_dir = os.path.join(save_dir, directory)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)

    # label信息保存
    txtfile = open(os.path.join(save_dir, 'label.txt'), mode='a')
    txtfile.writelines('%s %d \n' % (str(ii) + '.jpg', label1))
    txtfile.close()

    # z方向切片
    z_silc_50 = ArrayDicom[:, :, z_]
    z_silc_50 = cv2.resize(z_silc_50, (50, 50), interpolation=cv2.INTER_LINEAR) * 255
    z_silc_50 = z_silc_50.astype(np.uint8)  # 数据类型转换
    path = os.path.join(save_dir, 'z', str(ii) + '.jpg')
    cv2.imwrite(path, z_silc_50)

    # x方向切片
    x_silc_50 = ArrayDicom[x_, :, :]
    x_silc_50 = cv2.resize(x_silc_50, (50, 50), interpolation=cv2.INTER_LINEAR) * 255
    x_silc_50 = x_silc_50.astype(np.uint8)  # 数据类型转换
    cv2.imwrite(os.path.join(save_dir, 'x', str(ii) + '.jpg'), x_silc_50)

    # y方向切片
    y_silc_50 = ArrayDicom[:, y_, :]
    y_silc_50 = cv2.resize(y_silc_50, (50, 50), interpolation=cv2.INTER_LINEAR) * 255
    y_silc_50 = y_silc_50.astype(np.uint8)  # 数据类型转换
    cv2.imwrite(os.path.join(save_dir, 'y', str(ii) + '.jpg'), y_silc_50)

    ii += 1
    print(ii)
