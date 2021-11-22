import os
import cv2
import json
import glob
import numpy as np 
import argparse



parser = argparse.ArgumentParser(
    description="Mask generator of images from json file Semantic Segmentation")
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to directory where images and .json files are stored",
                    default="./",
                    type=str)
parser.add_argument("-m",
                    "--mask_dir",
                    help="Path were the generated masks should be stored",
                    default="./",
                    type=str)
parser.add_argument("-t",
                    "--text_file",
                    help="Path to text file contatining list of classes",
                    type=str)


args = parser.parse_args()

def get_classes(path):
  f = open(path , "r")
  classes = f.readlines()
  classes = [c.rstrip() for c in classes]
  f.close()
  return classes

def create_mask(json_path) :
  with open(json_path) as f:
    data = json.load(f)
  f.close()

  w,h = data["imageWidth"] , data["imageHeight"]
  image = data["imagePath"].split(".")[0]

  mask = np.zeros((h,w) ,dtype=np.uint8)


  for x in data["shapes"] :
      contours = np.array(x["points"] ,dtype=np.int32)
      cv2.fillPoly(mask, pts = [contours], color =(c_dict[x["label"]]) )
  print("Classes present in mask",image," : " , np.unique(mask))
  cv2.imwrite(os.path.join( mask_dir,image+".png"), mask)



labels_path = args.text_file
image_dir = args.image_dir
mask_dir = args.mask_dir

c = get_classes(labels_path)
c_dict = {k:int(v) for v, k in enumerate(c)}

images = glob.glob( os.path.join(image_dir, '*.jpg') )
print(images)
for image in images :
  json_path = image.split(".")[0]
  json_path = json_path+".json"
  create_mask(json_path)


