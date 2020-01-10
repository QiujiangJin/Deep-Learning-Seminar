import PIL
import torch
from torchvision.transforms import transforms
import numpy as np
from .network_LSTM import Encoder
from .network_LSTM import Binarizer
from .network_LSTM import Decoder

rnn_num = 16

def encode(img, bottleneck):

    transform = transforms.Compose([transforms.ToTensor()])
    inputs = transform(img).unsqueeze(0)

    encoder = Encoder()
    binarizer = Binarizer(int(bottleneck/512))
    decoder = Decoder(int(bottleneck/512))

    if torch.cuda.is_available():
        encoder = Encoder().cuda()
        binarizer = Binarizer(int(bottleneck/512)).cuda()
        decoder = Decoder(int(bottleneck/512)).cuda()
        encoder.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/encoder'))
        binarizer.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/binarizer'))
        decoder.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/decoder'))
    else:
        encoder.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/encoder', map_location=torch.device('cpu')))
        binarizer.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/binarizer', map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/decoder', map_location=torch.device('cpu')))

    encoder.eval()
    binarizer.eval()
    decoder.eval()
    
    if torch.cuda.is_available():
        e_1 = (torch.zeros(1, 256, 64, 64).cuda(), torch.zeros(1, 256, 64, 64).cuda())
        e_2 = (torch.zeros(1, 512, 32, 32).cuda(), torch.zeros(1, 512, 32, 32).cuda())
        e_3 = (torch.zeros(1, 512, 16, 16).cuda(), torch.zeros(1, 512, 16, 16).cuda())
        d_1 = (torch.zeros(1, 512, 16, 16).cuda(), torch.zeros(1, 512, 16, 16).cuda())
        d_2 = (torch.zeros(1, 512, 32, 32).cuda(), torch.zeros(1, 512, 32, 32).cuda())
        d_3 = (torch.zeros(1, 256, 64, 64).cuda(), torch.zeros(1, 256, 64, 64).cuda())
        d_4 = (torch.zeros(1, 128, 128, 128).cuda(), torch.zeros(1, 128, 128, 128).cuda())
        residual = inputs.cuda()
    else:
        e_1 = (torch.zeros(1, 256, 64, 64), torch.zeros(1, 256, 64, 64))
        e_2 = (torch.zeros(1, 512, 32, 32), torch.zeros(1, 512, 32, 32))
        e_3 = (torch.zeros(1, 512, 16, 16), torch.zeros(1, 512, 16, 16))
        d_1 = (torch.zeros(1, 512, 16, 16), torch.zeros(1, 512, 16, 16))
        d_2 = (torch.zeros(1, 512, 32, 32), torch.zeros(1, 512, 32, 32))
        d_3 = (torch.zeros(1, 256, 64, 64), torch.zeros(1, 256, 64, 64))
        d_4 = (torch.zeros(1, 128, 128, 128), torch.zeros(1, 128, 128, 128))
        residual = inputs

    binary = []
    for t in range(rnn_num):
        e_result, e_1, e_2, e_3 = encoder(residual, e_1, e_2, e_3)
        b_result = binarizer(e_result)
        outputs, d_1, d_2, d_3, d_4 = decoder(b_result, d_1, d_2, d_3, d_4)
        residual = residual - outputs
        binary.append((torch.Tensor.cpu(b_result).detach().numpy().astype(np.int8) + 1)//2)
    binary = np.stack(binary, axis=0)
    binary = np.packbits(binary, axis=-1)
    return binary
    
def decode(x, bottleneck):

    decoder = Decoder(int(bottleneck/512))

    if torch.cuda.is_available():
        decoder = Decoder(int(bottleneck/512)).cuda()
        decoder.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/decoder'))
    else:
        decoder.load_state_dict(torch.load('./project/models/model_LSTM_' + str(bottleneck) + '/decoder', map_location=torch.device('cpu')))

    decoder.eval()

    if torch.cuda.is_available():
        d_1 = (torch.zeros(1, 512, 16, 16).cuda(), torch.zeros(1, 512, 16, 16).cuda())
        d_2 = (torch.zeros(1, 512, 32, 32).cuda(), torch.zeros(1, 512, 32, 32).cuda())
        d_3 = (torch.zeros(1, 256, 64, 64).cuda(), torch.zeros(1, 256, 64, 64).cuda())
        d_4 = (torch.zeros(1, 128, 128, 128).cuda(), torch.zeros(1, 128, 128, 128).cuda())
    else:
        d_1 = (torch.zeros(1, 512, 16, 16), torch.zeros(1, 512, 16, 16))
        d_2 = (torch.zeros(1, 512, 32, 32), torch.zeros(1, 512, 32, 32))
        d_3 = (torch.zeros(1, 256, 64, 64), torch.zeros(1, 256, 64, 64))
        d_4 = (torch.zeros(1, 128, 128, 128), torch.zeros(1, 128, 128, 128))

    binary = np.unpackbits(x, axis=-1)
    binary = torch.from_numpy(binary).float()*2 - 1
    result = torch.zeros(1, 3, 256, 256)
    for t in range(rnn_num):
        outputs, d_1, d_2, d_3, d_4 = decoder(binary[t], d_1, d_2, d_3, d_4)
        result = result + outputs
    toImage = transforms.ToPILImage()
    image = toImage(result.squeeze())

    return image