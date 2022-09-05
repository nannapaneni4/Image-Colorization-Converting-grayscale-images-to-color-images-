import torch
import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
  def __init__(self, input_size=256):
    super(Net, self).__init__()
    

    # ResNet - First layer accepts grayscale images, 
    # and we take only the first few layers of ResNet for this task
    resnet = models.resnet18(num_classes=100)
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
    RESNET_FEATURE_SIZE = 128
    ## Upsampling Network
    self.upsample = nn.Sequential(     
      nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output

#class convdown():
  #def __init__(self,in_channels, out_channels, kernel, apply_batch_norm = True) -> None:
      #super().__init__()
      #if apply_batch_norm:
        #self.layer = nn.Sequential(
        #nn.Conv2d(in_channels, out_channels, kernel, padding=1, stride=2),
        #nn.BatchNorm2d(out_channels),
        #nn.LeakyReLU(),
      #)
      #else:
        #self.layer = nn.Sequential(
        #nn.Conv2d(in_channels, out_channels, kernel, padding=1, stride=2),
        #nn.LeakyReLU(),
      #)

class U2Net(Net):
  def __init__(self, input_size=256):
    super(U2Net, self).__init__()
    #downsampling
    self.downlayer0 = nn.Sequential(nn.Conv2d(1, 128, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.MaxPool2d(2))
    self.downlayer1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.MaxPool2d(2)) 
    self.downlayer2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),nn.MaxPool2d(2)) 
    self.downlayer3 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, stride=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),nn.MaxPool2d(2))  
    self.downlayer4 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, stride=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),nn.MaxPool2d(2))
    #upsampling
    self.uplayer1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.Upsample(scale_factor=2))
    self.uplayer2 = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.Upsample(scale_factor=2))
    self.uplayer3 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.Upsample(scale_factor=2))
    self.uplayer4 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.Upsample(scale_factor=2))
    self.uplayer5 = nn.Sequential(nn.Conv2d(256, 3, 3, padding=1, stride=1),
        nn.LeakyReLU(),nn.Upsample(scale_factor=2))
    self.outlayer =  nn.Conv2d(4,3,kernel_size=3, padding=1)

  
  
    
  #def convup(self,in_channels, out_channels, kernel):
    #return nn.Sequential(
      #nn.ConvTranspose2d(in_channels, out_channels, kernel, padding=1, stride=2),
      #nn.LeakyReLU(),
    #)
  """
  def forward(self, input):
    dlayer0 = self.convdown(1,128,3)(input)
    dlayer1 = self.convdown(128,128,3)(dlayer0)
    dlayer2 = self.convdown(128,256,3,True)(dlayer1)
    dlayer3 = self.convdown(256,512,3,True)(dlayer2)
    dlayer4 = self.convdown(512,512,3,True)(dlayer3)
    ulayer1 = self.convup(512,512,3)(dlayer4)
    c1 = torch.cat([ulayer1,dlayer4])
    ulayer2 = self.convup(1024,256,3)(c1)
    c2 = torch.cat([ulayer2,dlayer3])
    ulayer3 = self.convup(768,128,3)(c2)
    c3 = torch.cat([ulayer3,dlayer2])
    ulayer4 = self.convup(384,128,3)(c3)
    c4 = torch.cat([ulayer4,dlayer1])
    ulayer5 = self.convup(256,3,3)(c4)
    c5 = torch.cat([ulayer5,input])
    output = nn.Conv2d(6,3,kernel_size=2)(c5)
    return output
    """

  def forward(self,input):
    out_d0 = self.downlayer0(input)
    out_d1 = self.downlayer1(out_d0)
    out_d2 = self.downlayer2(out_d1)
    out_d3 = self.downlayer3(out_d2)
    out_d4 = self.downlayer4(out_d3)
    out_u1 = self.uplayer1(out_d4)
    out_c1 = torch.cat([out_d3,out_u1],dim=1)
    out_u2 = self.uplayer2(out_c1)
    out_c2 = torch.cat([out_d2,out_u2],dim=1)
    out_u3 = self.uplayer3(out_c2)
    out_c3 = torch.cat([out_u3,out_d1],dim=1)
    out_u4 = self.uplayer4(out_c3)
    out_c4 = torch.cat([out_u4,out_d0],dim=1)
    out_u5 = self.uplayer5(out_c4)
    out_c5 = torch.cat([out_u5,input],dim=1)
    output = self.outlayer(out_c5)
    return output
