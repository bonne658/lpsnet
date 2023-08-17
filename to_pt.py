import torch, sys
from collections import OrderedDict
from lpsnet import get_lspnet_s, get_lspnet_m

net=get_lspnet_m(2)
paras = torch.load('result/model/14572--0.04002.pth', map_location="cpu")
net.eval()
dct = OrderedDict()
for key in paras:
	if not 'aux' in key: dct[key]=paras[key]
#sys.exit()
net.load_state_dict(dct)
x = torch.randn((1, 3, 512, 1280))
traced_script_module = torch.jit.trace(net, x)
traced_script_module.save('lspnet.pt')
