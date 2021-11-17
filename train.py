import gluoncv
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from segdataset import VOCSegmentation
# List of all classes in the dataset
c=["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool",
   "vegetation","roof","wall","window","door","fence","fence-pole","person",
   "dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]


# Transforms for Normalization
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

# Create Dataset
trainset = VOCSegmentation(root='/kaggle/input/d/ninad08/segdata2/SegmentationDataset' ,split='train',
                            transform=input_transform ,cls=c)

# Create Training Loader
train_data = gluon.data.DataLoader(
    trainset, 12, shuffle=True, last_batch='rollover',
    num_workers=4)

print(len(trainset))

for i, (data, target) in enumerate(train_data):
        print(data.shape)
        print(target.shape)
        break