import torch
import torchvision
import os
import glob
import time
import pickle
import cv2
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data import LungDataset, blend, Pad, Crop, Resize
from src.models import UNet, PretrainedUNet
from src.metrics import jaccard, dice

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

models_folder = Path("/data/models/CV1")
images_folder = Path("/data/Segmentation_PADCHEST/CV1/images")
batch_size = 4

# Train Folder
Trdata_folder = Path("/data/Segmentation_PADCHEST/CVData_PADCHEST/CV1/", "train")
trorigins_folder = Trdata_folder / "img"
trmasks_folder = Trdata_folder / "seg"
print(trorigins_folder)

# Data loading
trorigins_list = [f.stem for f in trorigins_folder.glob("*.png")]
trmasks_list = [f.stem for f in trmasks_folder.glob("*.png")]

print("Len of trorigins_list: " , len(trorigins_list))
print("Len of trmasks_list: " , len(trmasks_list))

trorigin_mask_list = [(mask_name.replace("_segm", ""), mask_name) for mask_name in trmasks_list]

splits = {}
splits["train"] = trorigin_mask_list
splits["train"], splits["val"] = train_test_split(splits["train"], test_size=0.1, random_state=42)

print("Len of splits['train']: " , len(splits['train']))
print("Len of splits['val']: " , len(splits['val']))

# Train data

datasets = {x: LungDataset(
    splits[x],
    trorigins_folder,
    trmasks_folder) for x in ["train", "val"]}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size) for x in ["train", "val"]}

idx = 0
phase = "train"
#plt.figure(figsize=(20, 10))
origin, mask = datasets[phase][idx]
pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
pil_mask = torchvision.transforms.functional.to_pil_image(mask.float())
plt.subplot(1, 3, 1)
#plt.title("origin image")
plt.imshow(np.array(pil_origin))
plt.subplot(1, 3, 2)
plt.title("manually labeled mask")
plt.imshow(np.array(pil_mask))
plt.subplot(1, 3, 3)
plt.title("blended origin + mask")
plt.imshow(np.array(blend(origin, mask)));
plt.savefig(images_folder / "data-example.png", bbox_inches='tight')


# Model training
# unet = UNet(in_channels=1, out_channels=2, batch_norm=True)
unet = PretrainedUNet(
    in_channels=1,
    out_channels=2,
    batch_norm=True,
    upscale_mode="bilinear"
)
print(unet)

unet = unet.to(device)
# optimizer = torch.optim.SGD(unet.parameters(), lr=0.0005, momentum=0.9)
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0005)

train_log_filename = "train-log.txt"
epochs = 100
best_val_loss = np.inf
model_name = "unet-6v.pt"

hist = []

for e in range(epochs):
    start_t = time.time()

    print("train phase")
    unet.train()
    train_loss = 0.0
    for origins, masks in dataloaders["train"]:
        num = origins.size(0)

        origins = origins.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outs = unet(origins)
        softmax = torch.nn.functional.log_softmax(outs, dim=1)
        loss = torch.nn.functional.nll_loss(softmax, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * num
        print(".", end="")

    train_loss = train_loss / len(datasets['train'])
    print()

    print("validation phase")
    unet.eval()
    val_loss = 0.0
    val_jaccard = 0.0
    val_dice = 0.0

    for origins, masks in dataloaders["val"]:
        num = origins.size(0)

        origins = origins.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outs = unet(origins)
            softmax = torch.nn.functional.log_softmax(outs, dim=1)
            val_loss += torch.nn.functional.nll_loss(softmax, masks).item() * num

            outs = torch.argmax(softmax, dim=1)
            outs = outs.float()
            masks = masks.float()
            val_jaccard += jaccard(masks, outs.float()).item() * num
            val_dice += dice(masks, outs).item() * num

        print(".", end="")
    val_loss = val_loss / len(datasets["val"])
    val_jaccard = val_jaccard / len(datasets["val"])
    val_dice = val_dice / len(datasets["val"])
    print()


    end_t = time.time()
    spended_t = end_t - start_t

    with open(train_log_filename, "a") as train_log_file:
        report = f"epoch: {e+1}/{epochs}, time: {spended_t}, train loss: {train_loss}, \n"\
               + f"val loss: {val_loss}, val jaccard: {val_jaccard}, val dice: {val_dice}"

        hist.append({
            "time": spended_t,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_jaccard": val_jaccard,
            "val_dice": val_dice,
        })

        print(report)
        train_log_file.write(report + "\n")

        torch.save(unet.state_dict(), models_folder / model_name.replace(".pt", "_" + str(e+1) + "th_epoch.pt"))
        
        print("model saved")
        train_log_file.write("model saved\n")

        plt.figure(figsize=(15,7))
        train_loss_hist = [h["train_loss"] for h in hist]
        plt.plot(range(len(hist)), train_loss_hist, "b", label="train loss")

        val_loss_hist = [h["val_loss"] for h in hist]
        plt.plot(range(len(hist)), val_loss_hist, "r", label="val loss")

        val_dice_hist = [h["val_dice"] for h in hist]
        plt.plot(range(len(hist)), val_dice_hist, "g", label="val dice")

        val_jaccard_hist = [h["val_jaccard"] for h in hist]
        plt.plot(range(len(hist)), val_jaccard_hist, "y", label="val jaccard")

        plt.legend()
        plt.xlabel("epoch")
        plt.savefig(images_folder / model_name.replace(".pt", "_" + str(e+1) + "th_epoch-train-hist.png"))

        time_hist = [h["time"] for h in hist]
        overall_time = sum(time_hist) // 60
        mean_epoch_time = sum(time_hist) / len(hist)
        print(f"epochs: {len(hist)}, overall time: {overall_time}m, mean epoch time: {mean_epoch_time}s")

        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    torch.save(unet.state_dict(), models_folder / model_name)
        #    print("model saved")
        #    train_log_file.write("model saved\n")

        print()


plt.figure(figsize=(15,7))
train_loss_hist = [h["train_loss"] for h in hist]
plt.plot(range(len(hist)), train_loss_hist, "b", label="train loss")

val_loss_hist = [h["val_loss"] for h in hist]
plt.plot(range(len(hist)), val_loss_hist, "r", label="val loss")

val_dice_hist = [h["val_dice"] for h in hist]
plt.plot(range(len(hist)), val_dice_hist, "g", label="val dice")

val_jaccard_hist = [h["val_jaccard"] for h in hist]
plt.plot(range(len(hist)), val_jaccard_hist, "y", label="val jaccard")

plt.legend()
plt.xlabel("epoch")
plt.savefig(images_folder / model_name.replace(".pt", "LAST-train-hist.png"))

time_hist = [h["time"] for h in hist]
overall_time = sum(time_hist) // 60
mean_epoch_time = sum(time_hist) / len(hist)
print(f"epochs: {len(hist)}, overall time: {overall_time}m, mean epoch time: {mean_epoch_time}s")

























#
