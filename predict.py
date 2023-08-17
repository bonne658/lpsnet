import os, sys, cv2
import numpy as np
import torch, glob, time
from collections import OrderedDict
from torch.utils.data import DataLoader
from data import LWDDataset
from RepLPS import RepLPS
from lpsnet import get_lspnet_s, get_lspnet_m

bs = 1
# data
jpgs = glob.glob('/home/hookii/lwd/data/lawn/test/*jpg')
jpgs.sort()
n_val=len(jpgs)
# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=get_lspnet_m(2)
model = RepLPS()
model_path='/home/hookii/lwd/code/lspnet/result/model/13058--0.05676.pth'
paras=torch.load(model_path, map_location='cuda')
if 'state_dict' in paras: paras=paras['state_dict']
model.load_state_dict(paras)
model.to(device=device)
model.eval()

#criteria_pre = CELoss()
#palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
tm=time.time()
error=0
for jpg in jpgs:
	#im = cv2.imread('/home/lwd/data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')[:, :, ::-1]
	#im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
	im = cv2.imread(jpg)
	h,w,c=im.shape
	im = im.transpose(2, 0, 1).astype(np.float32).reshape(1,c,h,w)
	im /= 255.0
	im = torch.from_numpy(im).cuda()
	logits = model(im)
	#lb = lb.cuda()
	#loss_pre = criteria_pre(logits, lb)
	#print(loss_pre.item())
	res = logits.argmax(dim=1)
	res = res.squeeze().cpu().numpy().astype('uint8')#.transpose(1,0)
	#res=palette[res]
	res[res>0] = 255
	im=im[0]*255.0
	im = im.permute(1, 2, 0).cpu().numpy().astype('uint8')
	tmp = np.zeros(im.shape).astype('uint8')
	tmp[..., 2] = res/2
	im[res>0] = im[res>0] / 2 + tmp[res>0]
	#cv2.imshow('ss', im)
	#if cv2.waitKey() & 0xff == 27: break
	name = jpg.split("/")[-1]
	print(name)
	cv2.imwrite("result/image/"+name, im)
	png=jpg.replace('jpg', 'png')
	mask = cv2.imread(png, 0)
	mask=(mask!=res).sum()
	error+=mask
tm=time.time()-tm
print(tm/n_val)
print(error/n_val)
