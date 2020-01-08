#!/usr/bin/env python
# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function    
import torch                        
import torch.nn as nn              
import torch.nn.functional as F     
import torch.optim as optim         
from PIL import Image               
import matplotlib.pyplot as plt     
import torchvision.transforms as transforms    
import torchvision.models as models            # 常用模型
import copy                          
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#判断是否有GPU
#图片加载选项
imsize = 512 if torch.cuda.is_available() else 128     # GPU存在用512的图像否则用128
#图片裁剪
loader = transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])          
#图片装载
def image_loader(image_name):              
    image = Image.open(image_name)         
    image = loader(image).unsqueeze(0)     
    return image.to(device, torch.float)   
style_img = image_loader(r'D:\test\t1.jpg')      # 风格图
content_img = image_loader(r'D:\test\t2.jpg')    # 内容图
#判断两张图是否一样大
assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"  
#画出输入的两种张图片
unloader = transforms.ToPILImage()  # 将tensor转换为PIL
plt.ion()          # 开启PLT绘图的交互模式
def imshow(tensor, title=None):  # 定义图像输出函数
    image = tensor.cpu().clone()
    image = image.squeeze(0)      
    image = unloader(image)       
    plt.imshow(image)   
    if title is not None:   
        plt.title(title)
    plt.pause(0.001)    
plt.figure()                                
imshow(style_img, title='Style Image')      
plt.figure()
imshow(content_img, title='Content Image')
# 内容损失类
class ContentLoss(nn.Module):             
    def __init__(self, target,):               
        super(ContentLoss, self).__init__()    
        self.target = target.detach()         # 将target从计算图中分离出来，使其不具备梯度
    def forward(self, input):                        # 前馈方法
        self.loss = F.mse_loss(input, self.target)   # 算出均方误差
        return input
#gram矩阵
def gram_matrix(input):          # gram用于保存图像的风格
    a, b, c, d = input.size()    
    features = input.view(a * b, c * d) 
    G = torch.mm(features, features.t())  # 计算gram 内积；   
    return G.div(a * b * c * d)     # 归一化，除数为神经元个数。
# 风格损失 
class StyleLoss(nn.Module):     
    def __init__(self, target_feature):                     
        super(StyleLoss, self).__init__()                                                                               
        self.target = gram_matrix(target_feature).detach()   # 计算目标特征图的gram矩阵
    def forward(self, input):                      
        G = gram_matrix(input)                     # 计算input的gram矩阵
        self.loss = F.mse_loss(G, self.target)     # 使用mse度量目标风格的图片与输入图片之间的gram矩阵的mse损失
        return input
cnn = models.vgg19(pretrained=True).features.to(device).eval() #载入VGG19
#VGG模型训练所用的样本进行了规范化，这里需要进行规范化
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)    # 均值
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)     # 标准差
class Normalization(nn.Module):             
    def __init__(self, mean, std):
        super(Normalization, self).__init__()      
        self.mean = torch.tensor(mean).view(-1, 1, 1)    
        self.std = torch.tensor(std).view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std     #对样本进行归一化
#  选定几个卷积层进行计算
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,style_img, content_img,content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)   # 归一化模块
    content_losses = [] 
    style_losses = []      
    model = nn.Sequential(normalization)
    i = 0                    # 统计卷积层
    for layer in cnn.children():        
        if isinstance(layer, nn.Conv2d):       # 判断当前layer是不是nn.Conv2d类
            i += 1                          
            name = 'conv_{}'.format(i)              # 记录该层的名字
        elif isinstance(layer, nn.ReLU):       # 如果当前layer是不是nn.ReLU类
            name = 'relu_{}'.format(i)              # 记录该层的名字       
            layer = nn.ReLU(inplace=False) 
        elif isinstance(layer, nn.MaxPool2d):    
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):  
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))  

        model.add_module(name, layer)       # 将以上各层逐个添加到model这个模型中
        if name in content_layers:        
            target = model(content_img).detach()
            content_loss = ContentLoss(target)  
            model.add_module("content_loss_{}".format(i), content_loss) 
            content_losses.append(content_loss) #内容损失

        if name in style_layers:           
            # add style loss:
            target_feature = model(style_img).detach()   # 风格图片前馈
            style_loss = StyleLoss(target_feature)       # 风格损失
            model.add_module("style_loss_{}".format(i), style_loss)    
            style_losses.append(style_loss)              
    # 将最后一个风格或者内容层之后的所有层都剪除
    for i in range(len(model) - 1, -1, -1): 
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break                                                                 
    model = model[:(i + 1)]     
    return model, style_losses, content_losses  
# 输入样本
input_img = content_img.clone()      
#input_img = torch.randn(content_img.data.size(), device=device)#白噪声图片
plt.figure()
imshow(input_img, title='Input Image')
# 优化器
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])    
    return optimizer
# 训练函数 
def run_style_transfer(cnn, normalization_mean, normalization_std,  content_img, style_img, input_img, num_steps=300,style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)       # 调用子函数，构建模型
    optimizer = get_input_optimizer(input_img)                               # 优化器
    print('Optimizing..')
    run = [0]         # 迭代次数的计步器
    while run[0] <= num_steps:     # 迭代次数
        def closure():
            input_img.data.clamp_(0, 1)   # 每次对输入图片进行调整
            optimizer.zero_grad()    # 每次epoch的时候将梯度置为0
            model(input_img)        
            style_score = 0          # 本次epoch的风格损失
            content_score = 0

            for sl in style_losses:           
                style_score += sl.loss        #总的风格损失 
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight       # 风格损失乘以其权重
            content_score *= content_weight
            loss = style_score + content_score  # 总损失
            loss.backward()                     # 反馈
            run[0] += 1                         
            if run[0] % 20 == 0:                #打印结果
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score      # 总损失
        optimizer.step(closure)      # 优化
    input_img.data.clamp_(0, 1) 
    return input_img         # 输出结果
#主程序
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,content_img, style_img, input_img)               
plt.figure()
imshow(output, title='Output Image')      # 画出最终风格迁移后的图
plt.ioff()
plt.show()   