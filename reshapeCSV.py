import os
import numpy as np
import cv2
import pandas as pd
import codecs

#Read in data
df = pd.read_csv("/data/PHOSP/PHOSPMetadata.csv")
df.fillna(method = 'ffill',inplace = True)

segfolder="/data/train/seg/"
properFilePath=segfolder+df['image_name'][0]

#Transform the data into images and plot one to see
imgs = []
for i in range(0,len(df)):
    fileName=df['image_name'][i]
    #print(fileName)
    properFilePath = segfolder + fileName
    img = cv2.imread(properFilePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs.append(gray)

dotsList=[]
for i in range(0,len(df)):
    temp=[]
    temp2d=[]
    firstDot = df['ref_point_fst_lung'][i]
    secondDot = df['ref_point_snd_lung'][i]


    firstDot = firstDot.split("-")
    firstDot1 = firstDot[0].replace('(','').replace(')','')
    firstDot2 = firstDot[1].replace('(','').replace(')','')

    temp.append(int(firstDot1))
    temp.append(int(firstDot2))
    temp2d.append(temp)

    temp=[]

    secondDot = secondDot.split("-")
    secondDot1 = secondDot[0].replace('(','').replace(')','')
    secondDot2 = secondDot[1].replace('(','').replace(')','')

    temp.append(int(secondDot1))
    temp.append(int(secondDot2))
    temp2d.append(temp)

    dotsList.append(temp2d)

print("Number of images:",len(imgs))

import csv
cnt=0

with open("/data/PHOSP/PHOSPFinalVersion.csv", 'w') as outfile:
    w = csv.writer(outfile)

    w.writerow(['firstX','firstY','secondX','secondY','image_list'])

    for i in range(0,len(imgs)):

        fx=str(dotsList[i][0][0])
        fy=str(dotsList[i][0][1])
        sx=str(dotsList[i][1][0])
        sy=str(dotsList[i][1][1])

        stringg=""

        for row in imgs[i]:
            for itemIn in range(0,len(row)-1):
                stringg+=str(row[itemIn])+" "
            stringg+=str(row[itemIn+1])
            stringg+="\n"

        w.writerow([fx,fy,sx,sy,stringg])

        cnt+=1

        print(cnt)

        #if cnt==5:
        #    break












#
