import math
import helper
from helper import reverse_transform2



model.eval()   # Set model to evaluate mode


data_path = Path('data')
test_path= data_path/'val'/'images' ###cambiar a test3
test_file_names = np.array(sorted(list(test_path.glob('*.npy'))))


test_loader = make_loader(test_file_names, shuffle=True, transform=val_transform)

        
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)

pred = pred.data.cpu().numpy()
print(inputs.shape)
print(pred.shape)

input_images_rgb = [reverse_transform2(x) for x in inputs[:,:3,:,:].cpu()]
# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]
print(np.shape(input_images_rgb))
helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])