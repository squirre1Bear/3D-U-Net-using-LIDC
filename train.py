import time  # 监测训练用时
import os  # 文件操作
import logging
import random
from tqdm import tqdm
from utils import imgToTensor
from unet3D import UNet3D
from utils import metrics
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import warnings
# 这一条是用于忽略warning
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 指定TensorBoard路径，用于记录训练的指标
Summary_path = r'C:\Users\91604\Desktop\lungSeg\LIDC\runs'
if not os.path.exists(Summary_path):
    os.makedirs(Summary_path)
writer = SummaryWriter(log_dir=Summary_path, purge_step=0)

class MyTransform:
    def __init__(self, horizontal_flip_prob=0.5, rotation_angle=10):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_angle = rotation_angle

    def __call__(self, image, label):
            # image和label类型都是ndarray
            # 1. 水平翻转
            if random.random() < self.horizontal_flip_prob:
                image = np.flip(image, axis=-1)  # 一般情况下使用最后一个维度进行翻转
                label = np.flip(label, axis=-1)

            # 2. 随机旋转
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            image = np.rot90(image, k=int(angle // 90), axes=(1, 2)).copy()  # 旋转，axes根据要迁移的轴进行选择
            label = np.rot90(label, k=int(angle // 90), axes=(1, 2)).copy()

            # 3. 将ndarray转换为tensor
            # 注意需要将tensor形状弄成一样的（使用reshape()）
            image = transforms.ToTensor()(image).reshape(1, 72, 88, 104)
            label = transforms.ToTensor()(label).reshape(1, 72, 88, 104)

            return image, label

# train函数初始化了U-net，进行了选择运行设备、数据集导入与处理等基本操作。之后开始训练
# 下面传的参数confit是配置文件
def train(config):
    device = torch.device('cuda:0')
    # 直接用Model类中的unet
    model = UNet3D(in_channels=1)
    model.to(device)
    # 定义日志记录器
    logger = initLogger("3D_Unet")
    # 损失函数为交叉熵损失函数
    # 设置类别权重
    weights = torch.tensor([0.005, 1.31])  # 背景权重低，结节权重高
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # 对数据图像进行预处理
    transform = MyTransform()

    # 接下来读取图片和掩码，进行预处理
    dst_train = imgToTensor.UnetDataset(transform=transform)
    # 将上面方法的实例传入DataLoader，得到适合模型训练的数据类型
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config['batch_size'])

    dst_valid = imgToTensor.UnetDataset(transform=transform)
    dataloader_valid = DataLoader(dst_valid, batch_size=config['batch_size'], shuffle=False)

    # 准确率
    cur_acc = []
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=[config['momentum'],0.999], weight_decay=config['weight_decay'])

    # 循环训练
    for epoch in range(config['num_epoch']):
        epoch_start = time.time()
        model.train()  # 切换成训练模式

        # 初始化基本变量
        loss_sum = 0.0     # 总损失值
        correct_sum = 0.0      # 准确个数
        labeled_sum = 0.0  #
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        IoU = 0.0

        # 设置进度条。dataloader_train为迭代的对象, ncols为进度条长度
        tbar = tqdm(dataloader_train, ncols=120)

        # 迭代tbar。data是输入的图像，target为目标图像
        for batch_idx, (data, target) in enumerate(tbar):
            tic = time.time()

            # 梯度信息归零
            optimizer.zero_grad()
            # 使用model预测结果。输出的[batch_size,channel,D,H,W]是2*2*72*88**104的张量
            output = model(data)
            target = target.squeeze(1)

            # loss是个张量，使用.item()可以提取出对应的数值
            loss = criterion(output, target)
            loss_sum += loss.item()
            loss.backward()   # 反向传播
            optimizer.step()    # 更新模型参数

            # 计算所有像素点中预测正确的数目correct，预测中标记了的像素数，预测和原图之间的交集、并集
            correct, labeled, inter, unoin = metrics.eval_metrics(output, target, config['num_classes'])
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin

            # np.spaceing(1)是一个非常小的整数，用来防止“/0”的出现
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)    # 预测正确的像素数 / 预测了的所有像素
            IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)    # 交集数/并集数

            # 下面设置训练中输出显示哪些信息
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.4f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config['batch_size']),
                pixelAcc, IoU.mean(),
                time.time()-tic, time.time()-epoch_start))
            cur_acc.append(pixelAcc)

            # 将训练结果记录到log中
            writer.add_scalar('Train_loss', loss_sum/((batch_idx+1)*config['batch_size']),epoch)    # 写入当前损失值的一个函数关系
            writer.add_scalar('Train_Acc', pixelAcc, epoch)      # 写入准确率
            writer.add_scalar('Train_IOU', IoU.mean(), epoch)      # 写入交集大小 / 并集大小
            logger.info('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} IOU {}  mIoU {:.4f} '.format(
                epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                pixelAcc, toString(IoU), IoU.mean()))

        # 有效集
        test_start = time.time()
        max_pixACC = 0.0
        model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        mIoU = 0.0
        tbar = tqdm(dataloader_valid, ncols=100)


        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tbar):
                tic = time.time()

                # data, target = data.to(device), target.to(device)
                output = model(data)
                target = target.squeeze(1)
                loss = criterion(output, target)
                loss_sum += loss.item()

                correct, labeled, inter, unoin = metrics.eval_metrics(output, target, config['num_classes'])
                correct_sum += correct
                labeled_sum += labeled
                inter_sum += inter
                unoin_sum += unoin
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                tbar.set_description('VAL ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.4f} | bt {:.2f} et {:.2f}|'.format(
                    epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                    pixelAcc, mIoU.mean(),
                            time.time() - tic, time.time() - test_start))

            # 选择最佳模型的参数，使用准确率去进行比较
            if pixelAcc > max_pixACC:
                max_pixACC = pixelAcc
                if os.path.exists(config['save_model']['save_path']) is False:
                    os.mkdir(config['save_model']['save_path'])
                torch.save(model.state_dict(), os.path.join(config['save_model']['save_path'], 'unet.pth'))

        writer.add_scalar('Val_loss', loss_sum / ((batch_idx + 1) * config['batch_size']), epoch)
        writer.add_scalar('Val_Acc', pixelAcc, epoch)
        writer.add_scalar('Val_mIOU', mIoU.mean(), epoch)
        logger.info('VAL ({}) | Loss: {:.3f} | Acc {:.2f} IOU {} mIoU {:.4f} |'.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(mIoU), mIoU.mean()))


# IOU是记录了一批交并比的对象
def toString(IOU):
    result = '{'
    for i, num in enumerate(IOU):
        result += str(i) + ': ' + '{:.4f}, '.format(num)
    result +='}'
    return result

def initLogger(model_name):
    # 初始化log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = r'logs'
    if os.path.exists(log_path) is False:
        os.mkdir(log_path)
    # 重命名log文件
    log_name = os.path.join(log_path, model_name + '_'+ rq + '.log')
    log_file = log_name
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger