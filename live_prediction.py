import torch
from mish_cuda import MishCuda
from utils.general import non_max_suppression, output_to_target, plot_images_new
from helper import replace_mish_layers, revert_sync_batchnorm
import numpy as np
import time
import cv2
from picamera2 import *


# load and setup model
model = torch.load('/home/pi/ScaledYOLOv4/final_model.pt', map_location=torch.device('cpu'))
replace_mish_layers(model['model'], MishCuda, torch.nn.Mish())
model = revert_sync_batchnorm(model['model'])
model = model.float().fuse().eval()

picam2 = Picamera2()

capture_config = picam2.still_configuration()
picam2.configure(capture_config)

picam2.start()

# load image
image = picam2.switch_mode_and_capture_image(capture_config)
transposed = image.transpose(2, 0, 1)


# reformat image
contiguous = np.ascontiguousarray(transposed)
torch_img = torch.from_numpy(contiguous).to('cpu')
torch_float = torch_img.float()
torch_normalized = torch_float / 255.0
unsqueezed = torch_normalized.unsqueeze(0)

# get a prediction
start = time.perf_counter()
pred = model(unsqueezed)[0]
end = time.perf_counter()
print('prediction took %f seconds.', end - start)
non_maxed = non_max_suppression(pred)
output = output_to_target(non_maxed)[:1]
print(f"top prediction: {output}")
result = plot_images_new([torch_normalized], [output])

cv2.namedWindow("prediction")
cv2.imshow("prediction", result)
cv2.waitKey(0)
cv2.destroyWindow("prediction")

