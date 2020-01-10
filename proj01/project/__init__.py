import torch
from torchvision.transforms import transforms
from .model import Net
from PIL import Image
import torch.distributions as tdist

net = Net()
if torch.cuda.is_available():
    net = Net().cuda()
    net.load_state_dict(torch.load('./project/model_feature'))
else:
    net.load_state_dict(torch.load('./project/model_feature', map_location=torch.device('cpu')))
net.eval()

def superresolve(img, seed=2019):
    transform = transforms.Compose([transforms.ToTensor()])
    inputs = transform(img)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    outputs = net(inputs.unsqueeze(0)).squeeze()
    if torch.cuda.is_available():
        outputs = outputs.cuda()
    torch.manual_seed(seed)
    noise = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
    tmp = noise.sample(outputs.size()).squeeze(3)
    for i in range(3):
    	for j in range(128):
    		for k in range(128):
    			tmp[i][2*j][2*k] = 0.0
    outputs += tmp
    toImage = transforms.ToPILImage()
    res = toImage(outputs)

    return res
