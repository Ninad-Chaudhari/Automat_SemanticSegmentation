# Automate_SemanticSegmentation
Repository for Automating steps in training a semantic segmentation model on custom dataset(MXnet).

#### Link to colab tutorial : [Automate_SemanticSegmentation.ipynb](https://colab.research.google.com/drive/1uSmV2CUuxkmipl07AH3_AM6lw6TaWG4m?usp=sharing)
## Steps to Run :
* Clone the repo using following command :
```
git clone https://github.com/Ninad-Chaudhari/Automate_SemanticSegmentation.git
```
* Install the following dependencies :
 ```
pip install mxnet-cu101
pip install gluoncv
 ```
* Transfer all images in JPEGImages folder
### Preparing the dataset :
* For Semantic Segmentation first we need to create corresponding masks for images in `.json fomat`.
* We will use labelme for this task.
* To install labelme run the command :
```
pip install labelme
```
* After that type the following command in command prompt :
```
labelme
```
* This should open the following window :
![image](https://user-images.githubusercontent.com/65274398/142896290-4367003f-0af7-4aec-85cb-e3504360cbfa.png)
* Now click on `Open Dir` and open the directoy where images are stored (JPEGImages folder).
* Now click on Create Polygon and start creating the boundaries of objects and name the class for that polygon.
![image](https://user-images.githubusercontent.com/65274398/142896618-a4faa08a-f4a6-465c-8cd5-137646da7e65.png)
* Once done click on save this will create a .json file with the same image name in the same directory.
* After creating json file your image folder should look like :<br>
![image](https://user-images.githubusercontent.com/65274398/142897024-a73d770d-d654-4bb9-aeb3-728fc9f6e56b.png)
* Each image having its own .json file
* Now create a text file with list of all the classes in it.<br>
![image](https://user-images.githubusercontent.com/65274398/142897640-47fc95f6-81c4-4fcb-a3d3-57efb1c94bbd.png)
* Make sure that the `first line` is always `unlabeled` in your text file.
* Now run the command : 
```
python generate_mask.py -i [Path to JPEGImages folder] -m [Path to SegmentationClass folder] -t [Path to text file contatining classes]
```
* This will create the corresponding masks for your images in the SegmentationClass folder
* Now run the command :
  >* -tr : Traning split
  >* -t : Testing split
  >* -v : Validation split
  >* -r : Path to root cloned directory.
```
python setup.py -tr 80 -t 10 -v 10 -r /content/drive/MyDrive/Automate_SemanticSegmentation
```
* This will create `trainval.txt` , `val.txt` , `test.txt` files in ImageSets folder.
* Lets train our model by executing the command :
  >* -c : Path to text file containing list of classes.
  >* -p : Path to root cloned directory
  >* -b : Batch size (can  be changed based on the CPU/GPU memory you have for training)
  >* -lr : Learning rate
  >* -w : weight decay
  >* -e : Number of epochs
  >* -ch : Path to the checkpoint folder in the cloned directory 
```
python train.py -c /content/drive/MyDrive/Automate_SemanticSegmentation/labels.txt -p /content/drive/MyDrive/Automate_SemanticSegmentation -b 12 -lr 0.001 -w 0.0001 -e 10 -ch /content/drive/MyDrive/Automate_SemanticSegmentation/checkpoint
```
* Lets save our trained model by exporting .json and .params files :
  >* -e : Number of epoch for which we are saving the parameters
  >* -f : Path to .params files for that epoch
  >* -n : Total number of classes in the dataset.  
```
python save_model.py -e 10 -f /content/drive/MyDrive/Automate_SemanticSegmentation/checkpoint/epoch_0010.params -n 24
```
* For loading our model and predicting on test images follow the tutorial on this collab : [SemSegInference](https://colab.research.google.com/drive/1Yx3q3hjKzTWBeOU5lJhr3089pJj14v78?usp=sharing)
