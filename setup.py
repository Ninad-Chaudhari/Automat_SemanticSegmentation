import os
import argparse
import glob

PATH_ROOT = os.getcwd()
PATH_ROOT = PATH_ROOT.replace(" ","")

parser = argparse.ArgumentParser(
    description="Train test validation text file generator")
parser.add_argument("-tr",
                    "--train",
                    help="train split in percentage",
                    default=80,
                    type=int)
parser.add_argument("-t",
                    "--test",
                    help="Test split in percentage",
                    default=10,
                    type=int)
parser.add_argument("-v",
                    "--val",
                    help="Validation split in percentage",
                    type=int)
parser.add_argument("-r",
                    "--root",
                    default=10,
                    help="Path to root directory of dataset",
                    type=str)

args = parser.parse_args()

train_split = args.train
test_split = args.test
val_split = args.val
ROOT = args.root

mask_path = os.path.join(ROOT,'SegmentationClass')


files = glob.glob(mask_path+"/*.png")
trainval = open(os.path.join(ROOT,"ImageSets/trainval.txt"), "w")
test = open(os.path.join(ROOT,"ImageSets/test.txt"), "w")
val = open(os.path.join(ROOT,"ImageSets/val.txt"),"w")

total_rec = len(files)

split1 = (total_rec*train_split)//100
split2 = (total_rec*test_split)//100

tr=0
te=0
v=0
for f in files[:split1]:
    trainval.write(f.split(".")[0])
    tr+=1
    trainval.write("\n")
for f in files[split1:split1+split2]:
    test.write(f.split(".")[0])
    te+=1
    test.write("\n")
for f in files[split1 + split2:]:
    val.write(f.split(".")[0])
    v+=1
    val.write("\n")

print("Training records : " ,tr)
print("Testing records :",te)
print("Val records :",v )

test.close()
trainval.close()
val.close()

