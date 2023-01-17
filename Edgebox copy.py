import glob
import numpy as np

import torch

from torchvision import transforms

from PIL import Image

filelist = glob.glob('../SFSegNets/data/*.jpg')

print(len(filelist))
# On the other hand, if you know the file names already, just put them in a sequence:

# filelist = 'file1.bmp', 'file2.bmp', 'file3.bmp'

# Combining all the images into one numpy array

# To combine all the images into one array:
# x = np.array([np.array(Image.open(fname)) for fname in filelist])
to_tensor = transforms.ToTensor()

x = []
for fname in filelist:
    img = Image.open(fname)
    x.append(to_tensor(img))
batch = torch.stack(x)



# convert_tensor = transforms.ToTensor()
# x1 = convert_tensor(x)
print(batch.shape)