Pothole Detection for India Roads using GPS Tracking 
1. Download the Repository using the command
   1. git clone "https://github.com/Darshan2082002/pothole_detection_system.git" ( This will clone the repository)
   2. check the reposirty with your cloned one
2. To create the enivorment to run the  file
   1. To use conda(anaconda) best one
      1. conda create --name pothole python=3.9
      2. conda activate pothole
      3. conda env list ( To check the env is available in conda)
      4. pip install -r requirements.txt
    2. To use virtual env if your Visual studio code
       1. python -m venv venv (make sure you download the python 3.9 manaully in conda it does by itself)
       2. venv\Scripts\activate.bat (to activate the env) --> To do check the this link to for virtual enviorment setup for linux/windows/ MAC OS  ( This is official link--> https://docs.python.org/3/library/venv.html , This is Geek for Geek for better understanding --> https://www.w3schools.com/python/python_virtualenv.asp)
3. To download the dataset
   1.  Kaggle --> https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset
   2.  Unzip the file
   3.  copy the image and label(annotation) to dataset/raw/
4. To run split and process the data
   1.( Readme file update is pending it will be done within 2 days of time) --> In this repo you can directly deploy the model into raspberry pi and showcase the result to the panel and review
    


