import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import cv2
import pylidc as pl
from pylidc.utils import consensus
import os

dataset_path = r"C:\Users\91604\Desktop\CT肺结节分割\LIDC\manifest-1730534256759\LIDC-IDRI"
dicom_name = 'LIDC-IDRI-0001'
PathDicom = os.path.join(dataset_path, dicom_name)  # 构建当前DICOM文件夹的完整路径
# 查询当前病例的扫描数据，并将第一个扫描结果存储到scan变量中
# 使用pylidc库中query进行查询，查询的结果是第一个满足名字要求(dicom_name)的数据。返回结果是scan类型
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == dicom_name).first()
# 将scan类型转换为 3维数组
vol = scan.to_volume()   # 将该扫描数据转换成数组形式的体积（volume）
# 聚类注释（nodule annotations）以获取一组注释
# nods类型是list，打印出来是[[Annotation(id=84,scan_id=12), Annotation(id=85,scan_id=12), Annotation(id=86,scan_id=12), Annotation(id=87,scan_id=12)]]
# 注意是nods是 [[]]，要提取Annotation需要取nods[0]
nods = scan.cluster_annotations()   # 用cluster_annotations()方法，"聚类"(类似整合多个数据)当前扫描结果中的注释内容。

try:
    # anns类型是list，打印出来是[Annotation(id=84,scan_id=12), Annotation(id=85,scan_id=12), Annotation(id=86,scan_id=12), Annotation(id=87,scan_id=12)]
    # 由于0001有4个医生标注，所以打印出来有4个Annotation()
    anns = nods[0]   # 尝试获取第一个注释（annotation）
    # 遍历 anns 输出每个 annotation 的基本信息
    # anns[i]包含'Calcification', 'InternalStructure', 'Lobulation', 'Malignancy', 'Margin', 'Sphericity', 'Spiculation', 'Subtlety', 'Texture'等信息，使用dir(anns[0])可以查看包含的属性
    Malignancy = anns[0].Malignancy   # 获取注释中的恶性程度（Malignancy）信息
except IndexError:
   # 如果没有注释，或者无法获取第一个注释，则继续下一个DICOM文件夹
    exit(0)

# 执行共识合并（consensus consolidation）和50%的一致性水平（agreement level）。
# 我们在切片周围添加填充以提供上下文以进行查看。
# consensus()用于合并注释，生成共识的掩膜。
# anns是当前实例中提取出的注释。类型为list，包含多个字段
# 取出来的cmask是3维boole数组，大小是52*58*49，是结节的像素标记为True，否则为False。->这个是标准答案，是多个医生掩膜结果的平均值
# cbbox规定掩膜区域的边界框(consensus bounding box)，边界框是个长方体。z方向范围是66~115，长度是49
# masks记录各个医生标记的结果，是list，总共4个分量，每个分量是3维数组52*58*49。有四个医生标记时长度就为4
cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=[(0,0), (7,7), (20,20)])

# 提取相应的切片进行可视化
image = vol[cbbox]   # 用cbbox定义的边界框，从体积数据vol中提取切片
k = int(0.5 * (cbbox[2].stop - cbbox[2].start))    # 选中z方向上索引的中间值，用于选择要可视化的切片

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=1)
# 标记不同注释的边界
colors = ['r', 'g', 'b', 'y']
for j in range(len(masks)):    # 遍历所有的标注
    # find_contours()返回一个数组，包含所有轮廓点
    for c in find_contours(masks[j][:, :, k].astype(float), 0.5):
        label = "Annotation %d" % (j+1)
        plt.plot(c[:, 1], c[:, 0], colors[j], label=label)
# 绘制50%共识轮廓线
for c in find_contours(cmask[:, :, k].astype(float), 0.5):
    plt.plot(c[:, 1], c[:, 0], '--k', label='50% Consensus')
ax.axis('off')  # 关闭坐标轴
ax.legend()  # 显示图例
plt.tight_layout()  # 调整布局以适应图像
plt.show()  # 显示图像
