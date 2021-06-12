# BASEDNet Technical Documentation

## Table of Contents  
[Relevant Documentation](#relevant-documentation)  
[Commands](#commands)  
[Manually Annotating Data](#manually-annotating-data)  
[Preparing Data For Model 2](#preparing-data-for-model-2)  
[Run data.py](#run-data.py)  
[Run train.py](#run-train.py)  
[Train dhSegment (or use existing model)](#train-dhSegment)  
[Run grad_search.py](#run-grad-search.py)  
[Evaluate Accuracy of Predictions](#evaluate-accuracy-of-predictions)  


## Relevant Documentation

Training dhSegment: [https://hackmd.io/@ericke/B1G1fEJ9L#Training](https://hackmd.io/@ericke/B1G1fEJ9L#Training)  
Read The Docs: [https://dhsegment.readthedocs.io/en/latest/](https://dhsegment.readthedocs.io/en/latest/)  
GitHub: [https://github.com/dhlab-epfl/dhSegment](https://github.com/dhlab-epfl/dhSegment)  
What is dhSegment: [https://dhlab-epfl.github.io/dhSegment/](https://dhlab-epfl.github.io/dhSegment/)  

## Commands

Check if the gpu’s are available:  
```nvidia-smi``` 

**Running dhSegment**  
View current containers + status:  
```docker container ls -a```  
If the docker container doesn’t exist, create it:  

To run the docker:  
```docker start dhSegment```  
```docker exec -it dhSegment bash```  
```tensorboard --logdir .```  

Exit the docker using `ctrl d` and stop the container after use:  
```docker stop dhSegment```  

## Manually Annotating Data
Transkribus Labelling Tool can be found [here](https://transkribus.eu/Transkribus/).

## Preparing Data for Model 2
1. Collect all good y’s (xml files) into a single folder and the images for each y into another folder   
    a. For EEBO, images are at `jason/datasets/EEBO` and xml files are at `jason/datasets/EEBO/team_page`  
    b. For READ_BAD, the images are in the “input” folder at `jason/datasets/READ_BAD_MASTER/Train&Test Simple` and xml files are in the `page-gt` folder  
    c. **Note**: These folders can be found on the sakura server  
2. Run Aneesha & Lynn’s perturbation script    
    a. Found in `src/bad_baseline_generator.py` 
    b. Usage:  
    ```python bad_baseline_generator.py [input directory] [output directory]```  
    c. In this case, the input directory will be the good y’s from the dataset, and the output directory will contain bad y’s with perturbations applied  
3. Label both folders separately (this step can take awhile)  
    a. Found in `src/label_updated.py`  
    b. Usage:   
    ```  
    python label_updated.py [good y’s folder] [original .jpg images] [labeled_good] BASELINE  
    python label_updated.py [bad y’s folder] [original .jpg images] [labeled_bad] BASELINE  
    ```  
    c. Should now have two folders called `labeled_good` and `labeled_bad` with .png labeled images  
4. Now, split the folders into `train/val/test`  
    a. Download the split_folders package [here](https://pypi.org/project/split-folders/)
    b. Place `labeled_good` and `labeled_bad` as subdirectories of a new folder called `input`  
    c. Run split_folder command  
    ```  
    splitfolders --ratio .7 .2 .1 -- input  
    ```  
    d. You should now have a folder called `output` that contains test, train, and val data  

## Run data.py
1. data.py turns the images into numpy array representations that are labeled as good or bad y’s  
2. Usage:  

```  
python data.py  
```  

3. The input directory is the output directory from the last section that contains test, train, and val data  
4. The input directory is to be hard-coded by the individual who is programming (currently “model_data”)  

## Run train.py
1. This step trains Model 2  
2. Usage:  

```  
python train.py  
```  

3. The input directory is the output directory from the last section that contains test, train, and val numpy arrays  
4. The`/model_data` directory should contain 6 numpy arrays (2 Train, 2 Test, 2 Val) where each pair of numpy arrays is a pair of Training Labels and Binarized Baseline Prediction Images that have been resized to 1000x1000 pixels  

## Train dhSegment (or use existing model)  
1. Existing dhSegment models can be found at `projects/jason/dhSegment/READ_BAD/page_modelX` on the Sakura server  
    a. The best-performing pre-trained dhSegment model for our purposes was arbitrarily named ‘page_model4’  
2. Details on training a new dhSegment model can be found [here](https://hackmd.io/@ericke/B1G1fEJ9L#Training)  

## Run grad_search.py
1. This step gets the baseline predictions from the trained dhSegment and BASEDNet models  
2. Usage:  

```  
python grad_search.py  
```  

3. The input directory is the directory that contains test numpy arrays  
4. Similar to our other scripts, the grad_search script takes in the original .jpg images from our historical document archive. Within the script, the grad_search will first generate baseline predictions from the historical document, and with these predictions, BASEDNet will provide a holistic score for the overall quality of the baseline predictions. From this, given that a set of baselines is scored as ‘bad’ by BASEDNet, we optimize the baselines over several iterations of gradient-based optimization.  

## Evaluate Accuracy of Predictions
1. Download the following evaluation tool: [https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme](https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme)  
2. Usage (from inside the target folder):   

```   
java -jar TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar [truth.lst] [reco.lst]
```  

3. The .lst files should contains the list of names of the ground truth and output predicted files