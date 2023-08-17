from RepLPS import RepLPS
import torch

model = RepLPS()
model_path='result/model/27244--0.05208.pth'
paras=torch.load(model_path, map_location='cpu')
model.load_state_dict(paras)
for module in model.modules():
	if hasattr(module, 'switch_to_deploy'):
		module.switch_to_deploy()
torch.save(model.state_dict(), 'deploy.pth')
