import torch
from mish_cuda import MishCuda
from utils.general import non_max_suppression, output_to_target, plot_images_new
from helper import replace_mish_layers, revert_sync_batchnorm
import numpy as np
import time
import cv2

# load and setup model
model = torch.load('/home/pi/ScaledYOLOv4/final_model.pt', map_location=torch.device('cpu'))
replace_mish_layers(model['model'], MishCuda, torch.nn.Mish())
model = revert_sync_batchnorm(model['model'])
model = model.float().fuse().eval()

# load image
img_path = '/home/pi/ScaledYOLOv4/P1.png'
img = cv2.imread(img_path)
transposed = img[:, :, ::-1].transpose(2, 0, 1)


# reformat image
contiguous = np.ascontiguousarray(transposed)
torch_img = torch.from_numpy(contiguous).to('cpu')
torch_float = torch_img.float()
torch_normalized = torch_float / 255.0
unsqueezed = torch_float.unsqueeze(0)

# get a prediction
start = time.perf_counter()
pred = model(unsqueezed)[0]
end = time.perf_counter()
print('prediction took %f seconds.', end - start)
non_maxed = non_max_suppression(pred)
output = output_to_target(non_maxed)[:1]
print(f"top prediction: {output}")
result = plot_images_new([torch_normalized], [output])

