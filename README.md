# ZSPD-Deep-learning-project
<h1> Heart MRI Superresolution </h1>

<h2> Members of the ZSPD team: </h2>
<p> 
<ul>
  <li>Fris Domonkos - WPOU2Z</li>
  <li>Homolya Panni - ARQ7TQ</li>
  <li>Turcsik Zsófia - R7UJDO</li>
</ul>
</p>
<h2> Milestone 1. </h2>
<h3> Introduction </h3>

<p> Making high resolution MRI images takes quite a long time and the movement of the patient makes it difficult to make high quality images, but from the low resolution pictures it is harder to gather information about the subject.</p>

<h3> Our project </h3>

<p> In this project our objective is to convert low resolution heart MRIs to high resolution images with deep learning techniques. The data was downloaded from <a  href= 'https://acdc.creatis.insa-lyon.fr/description/databases.html'> data_link </a>. The dataset contains 100 patients folder, and each of the folder contains 5 files: data from the patient, 4D image, 2 frames image, 2 ground truth images. The 4D image and ground truth images are used.</p>

<p> The original data consists of 4 dimensional high resolution images of shape $(216,256,10,t)$, where $t$ denotes time and differs throughout the 100 patients, but it's usually around 30. We also have some ground truth files, which contains the segmentation of the heart on the MRIs.</p>

<p>With the <b> data_preparation.py </b> codes we reshaped the data and saved the 4D pictures to $10t$ different 2D images, then saved them to training, validation and test folders, still separating the data by patients. We also saved the values of the bounding boxes for each picture in a text file for each folder.</p>

<p> In the <b> data_preparation.py </b> code we prepared for converting the pictures to lower resolution images and prepared for segmentation. In the <b> data_discovering.ipynb </b> the original data and our most important functions are shown. </p>

<h3> Folders and files in our repository</h3>

<p> The data folder contains  the training, validation and test folders. Each folder contains the preprocessed image of the patient and a text file which will be used for modelling. Furthermore the <b> data_preparation.py </b> file contains the necessary functions for the jupyter notebook. The script and notebook are well commented.</p
