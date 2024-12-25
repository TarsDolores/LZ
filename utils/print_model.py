from models import my_Unet
from models import my_Unet_FCM
import torch
from torchsummary import summary
import io
import sys
import segmentation_models_pytorch as smp


num_classes = 6
resolutions = [(252, 252), (126, 126), (56, 56), (28, 28)]
model = my_Unet_FCM(3, num_classes, [32, 64, 128, 256, 512], input_resolutions=resolutions)

# model = smp.Unet(
#           encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#           #encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#          in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#          classes=6,  # model output channels (number of classes in your dataset)
#      )


#捕获输出
original_stdout = sys.stdout  # 保存当前的stdout
with io.StringIO() as buf:
     sys.stdout = buf  # 将stdout重定向到buf
     input_data = torch.ones(1, 3, 256, 256)
     summary(model, input_data, device='cpu')
     #print(model)
     output = buf.getvalue()  # 获取buf中的内容
     sys.stdout = original_stdout  # 恢复原来的stdout

 # 将输出写入文本文件
with open('model_summary_FCM.txt', 'w') as f:
     f.write(output)