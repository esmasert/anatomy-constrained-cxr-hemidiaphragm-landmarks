import os
import os.path
import codecs
import pandas as pd
import glob

txtFileName = "/data/PHOSP/PHOSPTrainSegList.txt"
with open(txtFileName, 'r') as file:
	content = file.read()

img_files= glob.glob("/data/train/img/*.png")
print("Number of training images:",len(img_files))

csvFilesList=[
"/data/PHOSP/phosp_measurements.csv"
]

myNewCSVfile= "/data/PHOSP/PHOSPMetadata.csv"
the_file = codecs.open(myNewCSVfile, "a")#.write(u"\u1234")
the_file.write("image_name,ref_point_fst_lung,ref_point_snd_lung")
the_file.write("\n")

count=0
bcount=0

for csvFileName in csvFilesList:

	print(csvFileName)
	cnt=0

	csvfile = open(csvFileName, 'r')
	csvFileLines = csvfile.readlines()
	df = pd.read_csv(csvFileName)
	df.fillna(method = 'ffill',inplace = True)

	for line in csvFileLines:
	    linelist = line.split(",")

	    fileName = linelist[0]

	    if fileName == "image_name":
	        #print(line)
	        continue

	    #segfolder="/data/PadChest/segray/"
	    extension="_segm.png"
	    properFilePath = fileName + extension

	    firstColumn = df['ref_point_fst_lung'][cnt]
	    firstColumn = firstColumn.replace(", ", "-")

	    secondColumn = df['ref_point_snd_lung'][cnt]
	    secondColumn = secondColumn.replace(", ", "-")

	    #print(linelist[-1])
	    if properFilePath in content:
	        the_file.write(properFilePath + "," + firstColumn + "," + secondColumn)
	        the_file.write("\n")
	        count+=1
	    else:
	        bcount+=1

	    cnt+=1


the_file.close()

print("Total number of chosen lines among csv files: ", count)
print("Total number of not chosen lines among csv files: ", bcount)
print("Number of training images:",len(img_files))














#
