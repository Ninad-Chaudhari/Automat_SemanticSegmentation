import gluoncv
from gluoncv.utils.parallel import *
import argparse
import mxnet as mx
from gluoncv.utils import export_block


parser = argparse.ArgumentParser(
    description="Script for exporting model .json and .params file")
parser.add_argument("-e",
                    "--epoch",
                    help="Epoch for which we need to export",
                    default=0,
                    type=int)

parser.add_argument("-f",
                    "--file",
                    help="Path to .params file for epoch",
                    default="./",
                    type=str)
parser.add_argument("-n",
                    "--n_class",
                    help="Number of classes in the dataset",
                    default=0,
                    type=int)


args = parser.parse_args()

model = gluoncv.model_zoo.FCN(nclass=args.n_class, backbone='resnet101', height=256, width=256 , ctx=mx.gpu(0))
ctx_list = [mx.gpu(0)]
model = DataParallelModel(model, ctx_list)
model.module.load_parameters(args.file, ctx=ctx_list)
model.module.hybridize()

export_block('fcn_resnet102', model.module ,preprocess=None,layout='CHW')