import os
from mxnet import gluon
from mxnet import image, np, npx , gpu
from mxnet.gluon.data import dataset
import cv2
import random
import numpy as np
import mxnet as mx
from mxnet import cpu

__all__ = ['ms_batchify_fn', 'SegmentationDataset']


class VisionDataset(dataset.Dataset):
    """Base Dataset with directory checker.
    Parametes
    ----------
    root : str
        The root path of xxx.names, by default is '~/.mxnet/datasets/foo', where
        `foo` is the name of the dataset.
    """
    
    def __init__(self, root):
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir. Did you forget to initialize \
                         datasets described in: \
                         `https://cv.gluon.ai/build/examples_datasets/index.html`? \
                         You need to initialize each dataset only once.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)
    
    

    
class SegmentationDataset(VisionDataset):
    """Segmentation base class
    Parametes
    ----------
    root : str
        The root path where the dataset is located
    split: str
        train , test , val
    transform:
        Augmentations to be applied to the image
    crop_size:int
       Size to which image is to be cropped
    """
    def __init__(self, root, split, mode, transform, base_size=520):
        super(SegmentationDataset, self).__init__(root)
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        

    def _val_sync_transform(self, img, mask):
        
        ow = 256
        oh = 256
        dim=(ow , oh)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
    

    def _sync_transform(self, img, mask):
        
        # random mirror
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            
        #resizing parameters    
        ow = 256
        oh = 256
        dim=(ow , oh)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
        

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
    

    def _img_transform(self, img):
        return F.array(np.array(img), cpu(0))
    

    def _mask_transform(self, mask):
        return F.array(np.array(mask), cpu(0)).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

def ms_batchify_fn(data):
    """Multi-size batchify function"""
    if isinstance(data[0], (str, mx.nd.NDArray)):
        return list(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [ms_batchify_fn(i) for i in data]
    raise RuntimeError('unknown datatype')
    
    
    
"""Pascal VOC Semantic Segmentation Dataset."""

class VOCSegmentation(SegmentationDataset):
    """Segmentation base class
        Parametes
        ----------
        root : str
            The root path where the dataset is located
        split: str
            train , test , val
        transform:
            Augmentations to be applied to the image
        cls : list
            List of all classes present in the dataset
    """

    NUM_CLASS = 0
    CLASSES=()
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/voc'),
                 split='train', mode=None, transform=None,cls=[], **kwargs):
        
        super(VOCSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = root
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
       
        self.images = []
        self.masks = []
        
        self.CLASSES=tuple(cls)
        self.NUM_CLASS=len(cls)
        
       
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                if(not os.path.isfile(_image)):
                    _image = os.path.join(_image_dir, line.rstrip('\n')+".jpeg")
                if(not os.path.isfile(_image)):
                    _image = os.path.join(_image_dir, line.rstrip('\n')+".JPG")
                
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
            

    def __getitem__(self, index):
         
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        if self.mode == 'test':
            img = self._img_transform(img)
           
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
    
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
   
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return F.array(target, cpu(0))

    @property
    def classes(self):
        """Category names."""
        pass
       # return type(self).CLASSES