from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

img = Image.open("train//ants_image//0013035.jpg")
img_arr = np.asarray(img)
writer = SummaryWriter("logs")
writer.add_image("test",img_arr,1,dataformats="HWC")

# for i in range(100):
#     writer.add_scalar("y=x",i,i)
writer.close()
