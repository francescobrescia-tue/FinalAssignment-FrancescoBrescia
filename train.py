"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torchvision.transforms as transforms
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
import utils

IMAGE_HEIGHT = 256  # 1024 originally
IMAGE_WIDTH = 512  # 2048 originally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001

IGNORE_INDEX=255
VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,255]
NUM_CLASSES=19

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser

def train_step(model, data_loader, loss_fn, optimizer, device):
    """Train the model for one epoch"""
    train_loss = 0
    train_jaccard_fn = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255).to(device)
    train_jaccard_fn.reset()
    loop = tqdm(data_loader)

    for images, masks in loop:
        images = images.to(device=device)
        masks = (masks*255).long().squeeze()
        masks = utils.map_id_to_train_id(masks).to(device=device)

        # forward
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_jaccard_fn.update(predictions, masks)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    train_loss /= len(data_loader)
    return train_loss, train_jaccard_fn.compute()

def validation_step(model, data_loader, loss_fn, device):
    """Validate the model on a data set"""
    val_loss = 0
    val_jaccard_fn = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255).to(device)
    val_jaccard_fn.reset()
    
    with torch.inference_mode():
        for images, masks in data_loader:
            images = images.to(device=device)
            masks = (masks*255).long().squeeze()
            masks = utils.map_id_to_train_id(masks).to(device=device)

            predictions = model(images)
            loss = loss_fn(predictions, masks)
            val_loss += loss.item()
            val_jaccard_fn.update(predictions, masks)
    
        val_loss /= len(data_loader)
        return val_loss, val_jaccard_fn.compute()

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    best_jaccard_index = 0  

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="final-assignment-cityscapes",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Transformer",
        "dataset": "CityScapes",
        "epochs": NUM_EPOCHS,
        }
    )

    transform = transforms.Compose([
        transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # data loading
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=mask_transform)

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
    #dataset[0]     //pair image mask
    #dataset[0][0]  //image  dim [3, 1024, 2048] (channels, height, width) type  torch.Tensor
    #dataset[0][1]  //mask   dim [1, 1024, 2048] (channels, height, width) type  torch.Tensor

    # visualize example images

    # define model
    model = Model(img_ch=3, output_ch=NUM_CLASSES).to(DEVICE)

    # define optimizer and loss function (don't forget to ignore class index 255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(DEVICE)
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        train_loss, train_jaccard_score = train_step(model, train_loader, loss_fn, optimizer, DEVICE)
        wandb.log({'Train Loss': train_loss, 'Train Jaccard': train_jaccard_score})
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Jaccard: {train_jaccard_score:.4f}")
    
        # evaluate on test set
        model.eval()
        val_loss, val_jaccard_score = validation_step(model, validation_loader, loss_fn, DEVICE) 
        wandb.log({'Validation Loss': val_loss, 'Validation Jaccard': val_jaccard_score})
        print(f"Validation {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}, Validation Jaccard: {val_jaccard_score:.4f}")
        
        if val_jaccard_score > best_jaccard_index:
            best_jaccard_index = val_jaccard_score
            torch.save(model.state_dict(), 'model.pth')
            print("Best model saved with Jaccard Index:", best_jaccard_index)

    #wandb.finish()
    
if __name__ == "__main__":
    
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)