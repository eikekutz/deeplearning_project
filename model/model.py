import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.momentum=0.1

        self.pool = nn.MaxPool2d(2, 2,return_indices=True)
        self.upsample = nn.MaxUnpool2d(2,2)

        self.en1 = self.encode_block(3,64,2)
        self.en2 = self.encode_block(64,128,2)
        self.en3 = self.encode_block(128,256,3)
        self.en4 = self.encode_block(256,512,3)
        self.en5 = self.encode_block(512,512,3)
  
        self.de5 = self.decode_block(512,512,3)
        self.de4 = self.decode_block(512,256,3)
        self.de3 = self.decode_block(256,128,3)
        self.de2 = self.decode_block(128,64,2)
        self.de1 = self.decode_block(64,2,2,LL=True)

        self.dropout = nn.Dropout(p=0.5)

    def encode_block(self,input_channel, output_channel,no_layer):
        en_list = []
        en_list.extend([
                            nn.Conv2d(input_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        for i in range(no_layer):
            en_list.extend([
                            nn.Conv2d(output_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        en_list.append(nn.MaxPool2d(2, 2,return_indices=True))

        return nn.Sequential(*en_list)
    def decode_block(self,input_channel, output_channel,no_layer,LL=False):
        de_list = []

        for i in range(no_layer):
            de_list.extend([
                            nn.Conv2d(input_channel,input_channel,3,padding=1),
                            nn.BatchNorm2d(input_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        de_list.append(nn.Conv2d(input_channel,output_channel,3,padding=1))
        if not LL:
            de_list.extend([
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
                            ])
        return nn.Sequential(*de_list)
    def forward(self,x):

        #encode blocks
        size0 = x.size()

        x,idx1 = self.en1(x)
        size1 = x.size()
        
        x,idx2 = self.en2(x)
        size2 = x.size()

        x,idx3 = self.en3(x)
        size3 = x.size()
        x = self.dropout(x)

        x,idx4 = self.en4(x)
        size4 = x.size()
        x = self.dropout(x)

        x,idx5 = self.en5(x)
        x = self.dropout(x)

        #decoder blocks
        x = self.upsample(x,idx5,output_size=size4)
        x = self.de5(x)
        x = self.dropout(x)

        x = self.upsample(x,idx4,output_size=size3)
        x = self.de4(x)
        x = self.dropout(x)

        x = self.upsample(x,idx3,output_size=size2)
        x = self.de3(x)
        x = self.dropout(x)

        x = self.upsample(x,idx2,output_size=size1)
        x = self.de2(x)

        x = self.upsample(x,idx1,output_size=size0)
        x = self.de1(x)

        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.momentum = 0.01
        self.en1 = self.encode_block(3,64,1,drop_rate=0.1)
        self.en2 = self.encode_block(64,128,1,drop_rate=0.3)
        self.en3 = self.encode_block(128,256,1,drop_rate=0.5)
        self.en4 = self.encode_block(256,512,1,drop_rate=0.5)

        self.boNe = self.bottleneck(512,1024,1,drop_rate=0.5)

        self.de4 = self.decode_block(1024,512,1,drop_rate=0.5)
        self.de3 = self.decode_block(512,256,1,drop_rate=0.5)
        self.de2 = self.decode_block(256,128,1,drop_rate=0.3)
        self.de1 = self.decode_block(128,64,1,LL=True)

        self.pool = nn.MaxPool2d(2, 2)



    def encode_block(self,input_channel, output_channel,no_layer,drop_rate=0):
        en_list = []
        en_list.extend([
                            nn.Conv2d(input_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        for i in range(no_layer):
            en_list.extend([
                            nn.Conv2d(output_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        if drop_rate > 0:
            en_list += [nn.Dropout(drop_rate)]
        return nn.Sequential(*en_list)

    def bottleneck (self,input_channel, output_channel,no_layer,drop_rate=0):
        b_list = []
        b_list.extend([
                            nn.Conv2d(input_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        for i in range(no_layer):
            b_list.extend([
                            nn.Conv2d(output_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        if drop_rate > 0:
            b_list += [nn.Dropout(drop_rate)]
        return nn.Sequential(*b_list)
    def upsample_cat(self,left_input, bottom_input):
        upsample = nn.Upsample(int(bottom_input.shape[-1]*2))  # double bottom image dimensions
        bottom_input = upsample(bottom_input)
        return torch.cat([bottom_input,left_input],1)

    def decode_block(self,input_channel, output_channel,no_layer,LL=False,drop_rate=0):
        de_list = []
        de_list.extend([
                        nn.Conv2d(input_channel+int(input_channel*0.5),output_channel,3,padding=1),
                        nn.BatchNorm2d(output_channel,momentum=self.momentum),
                        nn.ReLU()
        ])
        for i in range(no_layer):
            de_list.extend([
                            nn.Conv2d(output_channel,output_channel,3,padding=1),
                            nn.BatchNorm2d(output_channel,momentum=self.momentum),
                            nn.ReLU()
            ])
        if drop_rate > 0:
            de_list += [nn.Dropout(drop_rate)]
        if LL:
            de_list.append(nn.Conv2d(output_channel,2,3,padding=1))
        return nn.Sequential(*de_list)
    def forward(self,x):
        e1 = self.en1(x)
        x = self.pool(e1)
        e2 = self.en2(x)
        x = self.pool(e2)
        e3 = self.en3(x) 
        x = self.pool(e3)
        e4 = self.en4(x)
        x = self.pool(e4)

        b = self.boNe(x)

        d4 = self.upsample_cat(e4,b)
        d4 = self.de4(d4)
        d3 = self.upsample_cat(e3,d4)
        d3 = self.de3(d3)
        d2 = self.upsample_cat(e2,d3)
        d2 = self.de2(d2)
        d1 = self.upsample_cat(e1,d2)
        d1 = self.de1(d1)

        return d1