import numpy as np
import torch

# 输入为预测的结果、实际的结果、预测类别总数
# 最初target中 0：背景；1：肺结节
def eval_metrics(output, target, num_classes):
    # max(1)是获得第1个维度的最大值。这里用于判断预测结果
    # print(output.shape) [2, 2, 72, 88, 104]
    _, predict = output.max(1)
    # 预测结果变为long。+1让像素标签变为1(背景)、2(肺结节)。防止计算时遗漏标签为0的像素点
    predict = predict.long() + 1
    target = target.long() + 1

    bac = (target == 1).sum()     # 总共1317626个背景
    lun = (target == 2).sum()     # 只有262个肺结节
    pre_bac = (predict == 1).sum()
    pre_lun = (predict == 2).sum()

    print("lun="+str(lun)+"  pre_lun="+str(pre_lun))

    # 求出图片中总的像素数。target>0是排除里面有误的数据
    pixel_labeled = (target>0).sum()
    pixel_correct = ((predict == target) * (target>0)).sum()

    # 求预测像素的大小
    predict = predict * (target>0).long()
    intersection = predict * (predict == target).long()

    '''
        下面用的torch.histc()用于计算一个张量的直方图（每个类别有多大的数量），有四个参数:
        input (Tensor) -输入张量。  
        Bins (int) -直方图Bins(分类/类别)的数量  
        Min (int) -整个分类范围的下端(包括在内)  
        Max (int) -分类范围的上端(包括在内)  
    '''
    # 计算直方图中每个类别的像素数量
    area_inter = torch.histc(intersection.float(), num_classes, 1, num_classes)
    area_pred = torch.histc(predict.float(), num_classes, 1, num_classes)
    area_lab = torch.histc(target.float(), num_classes, 1, num_classes)
    area_union = area_pred + area_lab - area_inter

    # np.round:四舍五入
    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    #pixacc = 1.0 * correct / (np.spacing(1) + labeld)
    #mIoU = (1.0 * inter / (np.spacing(1) + union)).mean()
    return correct, labeld, inter, union