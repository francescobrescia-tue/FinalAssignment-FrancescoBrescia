import torch
from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torchvision.transforms as transforms
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from loss import focal_loss
from torch.utils.data import DataLoader, random_split
import wandb
import utils

IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 512  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 250
BATCH_SIZE = 16
LEARNING_RATE = 0.001

IGNORE_INDEX = 255
NUM_CLASSES = 19

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    return parser

def train_step(model, data_loader, loss_fn, optimizer, device):
    train_loss = 0
    train_jaccard_fn = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(device)
    train_jaccard_fn.reset()
    loop = tqdm(data_loader)
    for images, masks in loop:
        images = images.to(device)
        masks = (masks*255).long().squeeze()
        masks = utils.map_id_to_train_id(masks).to(device)
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_jaccard_fn.update(predictions, masks)
        loop.set_postfix(loss=loss.item())
    train_loss /= len(data_loader)
    return train_loss, train_jaccard_fn.compute()

def validation_step(model, data_loader, loss_fn, device):
    val_loss = 0
    val_jaccard_fn = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(device)
    val_jaccard_fn.reset()
    with torch.inference_mode():
        for images, masks in data_loader:
            images = images.to(device)
            masks = (masks*255).long().squeeze()
            masks = utils.map_id_to_train_id(masks).to(device)
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            val_loss += loss.item()
            val_jaccard_fn.update(predictions, masks)
        val_loss /= len(data_loader)
        return val_loss, val_jaccard_fn.compute()

def main(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="final-assignment-cityscapes",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Attention_Unet_Focal",
        "dataset": "CityScapes",
        "epochs": NUM_EPOCHS,
        }
    )  

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    validation_transform = transforms.Compose([
        transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # Data loading with correct transformations applied
    full_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', 
                              transform=train_transform, target_transform=mask_transform)
    
    # Splitting dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
    
    # Ensuring each subset uses the correct transform
    train_dataset.dataset.transform = train_transform
    validation_dataset.dataset.transform = validation_transform


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    model = Model(output_ch=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    loss_fn = focal_loss(ignore_index=IGNORE_INDEX).to(DEVICE)

    best_jaccard_index = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_jaccard_score = train_step(model, train_loader, loss_fn, optimizer, DEVICE)
        wandb.log({'Train Loss': train_loss, 'Train Jaccard': train_jaccard_score})
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Jaccard: {train_jaccard_score:.4f}")

        model.eval()
        val_loss, val_jaccard_score = validation_step(model, validation_loader, loss_fn, DEVICE)
        wandb.log({'Validation Loss': val_loss, 'Validation Jaccard': val_jaccard_score})
        print(f"Validation {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}, Validation Jaccard: {val_jaccard_score:.4f}")

        if val_jaccard_score > best_jaccard_index:
            best_jaccard_index = val_jaccard_score
            torch.save(model.state_dict(), 'model.pth')
            print("Best model saved with Jaccard Index:", best_jaccard_index)

    wandb.finish()

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)