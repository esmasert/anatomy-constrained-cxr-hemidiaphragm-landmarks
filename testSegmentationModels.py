import sys
sys.path.insert(0,'/data/HeatMap/src')

import torch
import torchvision
import os
import glob
import time
import pickle
import cv2
import torchvision.transforms as transforms
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from data import LungDataset, blend, Pad, Crop, Resize
from models import UNet, PretrainedUNet
from metricsUp import jaccard, dice, auc, precision, recall, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device

batch_size = 4

DIR = '/data/test/'
origins_folder = DIR + "img/"
masks_folder = DIR + "seg/"
models_folder = "/data/testModels"
images_folder = "./test_result_images/"

mlist=os.listdir(DIR+"seg/")
#len(mlist)

PATHGLOB = Path(DIR).glob('./*.png')
LS = [fil for fil in PATHGLOB]
#len(LS)

origins_list = [f.stem for f in Path(origins_folder).glob("./*.png")]
masks_list = [f.stem for f in Path(masks_folder).glob("./*.png")]

print(len(origins_list))
print(len(masks_list))

origin_mask_list = [(mask_name.replace("_segm", ""), mask_name) for mask_name in masks_list]

#len(origin_mask_list)

masks_folder=Path(masks_folder)
origins_folder=Path(origins_folder)

splits = {}
splits["test"] = origin_mask_list

datasets = {x: LungDataset(
    splits[x],
    origins_folder,
    masks_folder) for x in ["test"]}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size) for x in ["test"]}

#len(datasets["test"])


#EVALUATE

unet = PretrainedUNet(1, 2, True, "bilinear")
model_name = "/data/testModels/segmentation_unet_100th_epoch.pt"
unet.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
unet.to(device)
unet.eval();

test_loss = 0.0
test_jaccard = 0.0
test_dice = 0.0
test_auc = 0.0
test_precision = 0.0
test_recall = 0.0
test_accuracy = 0.0

print("LENGTH OF TEST DATASET:",len(datasets["test"]))

resultCsvFile = "/data/SegmentationTestResults.csv"
columns = ['ImgName', 'jaccard', 'dice', 'auc', 'precision', 'recall', 'accuracy']
with open(resultCsvFile, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(columns)

count=0
print("ORGINS LENGTH:",len(origins))
for origins, masks in dataloaders["test"]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num = origins.size(0)
    origins = origins.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outs = unet(origins)
        softmax = torch.nn.functional.log_softmax(outs, dim=1)
        test_loss += torch.nn.functional.nll_loss(softmax, masks).item() * num

        outs = torch.argmax(softmax, dim=1)
        outs = outs.float()
        masks = masks.float()

        device = torch.device("cpu")

        instant_jaccard = jaccard(masks, outs).item()
        instant_dice = dice(masks, outs).item()
        instant_auc = auc(masks, outs)
        instant_precision = precision(masks, outs)
        instant_recall = recall(masks, outs)
        instant_accuracy = accuracy(masks, outs)

        test_jaccard += instant_jaccard * num
        test_dice += instant_dice * num
        test_auc += instant_auc
        test_precision += instant_precision
        test_recall += instant_recall
        test_accuracy += instant_accuracy

        results = [origins_list[count],instant_jaccard,instant_dice,instant_auc,instant_precision,instant_recall,instant_accuracy]
        with open(resultCsvFile, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(results)

        count+=1
        print(count)

        if count==10:
            break

    #print(".", end="")

test_loss = test_loss / count
test_jaccard = test_jaccard / count
test_dice = test_dice / count
test_auc = test_auc / count
test_precision = test_precision / count
test_recall = test_recall / count
test_accuracy = test_accuracy / count

print(f"avg test loss: {test_loss}")
print(f"avg test jaccard: {test_jaccard}")
print(f"avg test dice: {test_dice}")
print(f"avg test auc: {test_auc}")
print(f"avg test precision: {test_precision}")
print(f"avg test recall: {test_recall}")
print(f"avg test accuracy: {test_accuracy}")


num_samples = 9
phase = "test"

subset = torch.utils.data.Subset(
    datasets[phase],
    np.random.randint(0, len(datasets[phase]), num_samples)
)
random_samples_loader = torch.utils.data.DataLoader(subset, batch_size=1)
plt.figure(figsize=(20, 25))

for idx, (origin, mask) in enumerate(random_samples_loader):
    plt.subplot((num_samples // 3) + 1, 3, idx+1)

    origin = origin.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        out = unet(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)

        jaccard_score = jaccard(mask.float(), out.float()).item()
        dice_score = dice(mask.float(), out.float()).item()

        origin = origin[0].to("cpu")
        out = out[0].to("cpu")
        mask = mask[0].to("cpu")

        plt.imshow(np.array(blend(origin, mask, out)))
        plt.title(f"jaccard: {jaccard_score:.4f}, dice: {dice_score:.4f}")
        print(".", end="")

plt.savefig(images_folder + "obtained-results.png", bbox_inches='tight')
print("red area - predict")
print("green area - ground truth")
print("yellow area - intersection")
















#
