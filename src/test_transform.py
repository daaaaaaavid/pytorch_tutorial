from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
img = Image.open("train/ants_image/374435068_7eee412ec4.jpg")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("logs")
writer.add_image("tensor_img",tensor_img)
writer.close()