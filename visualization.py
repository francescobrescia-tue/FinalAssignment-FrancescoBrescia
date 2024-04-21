import torch
import model as m
import best_model as bm
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utils
import numpy as np
from utils import LABELS
import torchvision.transforms.functional as TF
from torchmetrics.classification import MulticlassJaccardIndex

IMAGE_HEIGHT = 256  # 1024 originally
IMAGE_WIDTH = 512  # 2048 originally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES=60

def id_to_color(img_tensor, label_info):
    # Ensure img_tensor is on CPU before converting to numpy
    if img_tensor.is_cuda:
        img_tensor = img_tensor.cpu()

    # Initialize a color image with the same spatial dimensions as img_tensor
    color_image = np.zeros((img_tensor.shape[0], img_tensor.shape[1], 3), dtype=np.uint8)

    for label in label_info:
        if label.trainId != 255:  # Ensure we're not trying to color 'ignored' labels
            mask = (img_tensor == label.trainId).numpy()  # Convert to numpy array here
            color_image[mask] = label.color

    return color_image

def create_color_to_id_mapping(labels):
    color_to_id = {label.color: label.id for label in labels if label.id != -1}
    return color_to_id

def rgb_image_to_class_id(image, color_to_id):
    # Convert image to numpy array if it's a PIL Image
    image_array = np.array(image)
    class_id_image = np.zeros(image_array.shape[:2], dtype=int)

    # Apply color to class ID mapping
    for color, class_id in color_to_id.items():
        mask = np.all(image_array == np.array(color, dtype=np.uint8), axis=-1)
        class_id_image[mask] = class_id

    return class_id_image

def visualize_mask_with_train_ids(train_id_mask, labels):
    # Initialize a blank RGB image
    height, width = train_id_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Map train IDs back to colors
    for label in labels:
        color_mask[train_id_mask == label.trainId] = label.color

    return color_mask

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up the model
    model = m.Model(out_channels=60).to(device)
    model.load_state_dict(torch.load('baseline_model.pth', map_location=device))
    model.eval()

    best_model = bm.Model(output_ch=60).to(device)
    best_model.load_state_dict(torch.load('improved_model.pth', map_location=device))
    best_model.eval()

    # Load original image and mask without any transformation
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic')
    original_img, original_mask = dataset[0]

    if not isinstance(original_mask, np.ndarray):
        original_mask = np.array(original_mask)  # Convert PIL Image to numpy array if necessary

    # Check the shape of the mask to confirm it's single-channel
    if original_mask.ndim != 2:
        raise ValueError(f"Mask is expected to be 2D, but got shape {original_mask.shape}")

    # Visualize the mask with training IDs (convert IDs if your LABELS use different mapping)
    train_id_mask = utils.map_id_to_train_id(torch.tensor(original_mask, dtype=torch.int32))
    mask_visualization = visualize_mask_with_train_ids(train_id_mask.numpy(), LABELS)

    # Display original image and mask
    plt.figure(figsize=(20, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mask_visualization)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # Transform the image for model input
    transform = transforms.Compose([
        transforms.Resize([256, 512]),  # Change these dimensions as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        predict = model(img_tensor)
        predict_best = best_model(img_tensor)

    prediction = torch.argmax(predict, dim=1)
    prediction_resized = transforms.functional.resize(prediction, size=original_img.size[::-1], interpolation=transforms.InterpolationMode.NEAREST)
    prediction_best = torch.argmax(predict_best, dim=1)
    prediction_resized_best = transforms.functional.resize(prediction_best, size=original_img.size[::-1], interpolation=transforms.InterpolationMode.NEAREST)

    # Ensure it is 2D
    if prediction_resized.dim() > 2:
        prediction_resized = prediction_resized.squeeze(0)
    if prediction_resized_best.dim() > 2:
        prediction_resized_best = prediction_resized_best.squeeze(0)

    # Now pass it to id_to_color
    prediction_color = id_to_color(prediction_resized, LABELS)
    prediction_color_best = id_to_color(prediction_resized_best, LABELS)



    # Ensure gt_tensor matches the prediction tensor's format
    gt_tensor = train_id_mask.clone().detach().long().to(device)  # Proper way to handle tensors
    # Assuming train_id_mask is already in the correct form

    # If prediction has an extra batch or singleton dimension, ensure to remove it
    prediction_resized = prediction_resized.squeeze(0)  # Adjust according to actual tensor shape
    prediction_resized_best = prediction_resized_best.squeeze(0)

    # Initialize the Jaccard index object
    jaccard_fn = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255).to(device)

    # For baseline model predictions
    jaccard_fn.reset()
    jaccard_fn.update(prediction_resized.unsqueeze(0), gt_tensor.unsqueeze(0))  # Add batch dimension if needed
    baseline_iou = jaccard_fn.compute()

    # For best model predictions
    jaccard_fn.reset()
    jaccard_fn.update(prediction_resized_best.unsqueeze(0), gt_tensor.unsqueeze(0))  # Ensure batch dimension is present
    best_model_iou = jaccard_fn.compute()



    plt.subplot(1, 4, 3)
    plt.imshow(prediction_color)
    plt.title(f"Baseline Model Prediction\nIoU: {baseline_iou:.4f}")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(prediction_color_best)
    plt.title(f"Improved Model Prediction\nIoU: {best_model_iou:.4f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    args = parser.parse_args()
    main(args)