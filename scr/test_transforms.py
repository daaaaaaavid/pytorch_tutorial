from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("train/ants_image/2265824718_2c96f485da.jpg")


#to tensor
trans_totensor = transforms.ToTensor()
tensor_img = trans_totensor(img)
writer.add_image("tensor img",tensor_img)

#normalize
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
norm_img = trans_norm(tensor_img)
writer.add_image("normalize img",norm_img)

#resize
trans_resize = transforms.Resize((512,512))
resize_img = trans_resize(img)
resize_tensor_img = trans_totensor(resize_img)
writer.add_image("resize img",resize_tensor_img)

#compose
trans_resize_2 = transforms.Resize((512,512))
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
resize_img_2 = trans_compose(img)
writer.add_image("resize+transform",resize_img_2,1)

writer.close()

