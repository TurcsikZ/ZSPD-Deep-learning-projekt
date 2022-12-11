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

<p> The data folder contains the raw_data,the train, validation and test folders. The raw_data folder contains the origanal data. The other folders contain the preprocessed image of the patient and a text file which will be used for modeling.</p>

<p> The 2D_DATA folder contains the 2D images which splitted as data folder. The GT_data folder contains the ground truth images which splitted as before. </p>

<p> The data_manipulation folder contains all the jupyter notebooks which we used for create our datasets. The script and notebooks are commented.</p>

<p> The models folder contains all the results and models code which we used for the training and validation. </p> 
<p> Documentation folder contains the abstracts and the documentation files.</p> 

<h2> Milestone 2. </h2>
<p> We used the <a  href= 'https://www.paperspace.com/'> paperspace </a> platform to run our models.It has an own pytorch enviroment. </p>
<h3> Training models </h3>
<h4>1.SRGAN model  <a  href= 'https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan'> SRGAN_citation </a></h4>
<h4>We are using the model in the cited github repo. </h4>
<h5>1.1 SRGAN training </h5>

<p> Steps: </p>
<p> 
<ol>
  <li>Download the 2D_DATA, GT_data folder and unzip the data</li>
  <li>Download the srgan folder</li>
  <li>Run the script in the following way: python srgan.py --dataset_name path/2D_DATA/TRAIN/*</li>
  <li>Using the same script to continue the training with pre-trained models </li>
</ol>
IMPORTANT: 
When the model is being retrained, we used --dataset_name path/GT_data/GT_TRAIN/* and we gave our model the saved discriminator and generator weights. Furthermore we used the following learning rates: [0.001, 0.0002, 0.00001]. We did not change other parameters.
</p>

<h5>1.2 SRGAN evaluation </h5>
<p> For the evaluation we run the models/srgan/evaluation/srgan_val.py file. For the first evaluation of the retrained models we used the GT_data/GT_VAl set. The evaluation of the final model we used the same py file with GT_data/GT_TEST set. </p>
<p>Run the script in the following way(dataset can be changed): python srgan_val.py --dataset_name path/GT_data/GT_VAl/* </p>
<p> The results of the evaluation are in the models/srgan/evaluation folder. The necessary files are the following: train_test_model.ipynb, val_test_model.ipynb, final_test.ipynb .</p>

<h4>2. 2D WGAN <a  href= 'https://github.com/Hadrien-Cornier/E6040-super-resolution-project'> WGAN_citation </a> </h4>
<h4>This is basically the same model used in the citation, but we modified it, so it works with 2D single images. </h4>
<h5>2.1 Training </h5>

<p> In the folder there is a test_train.pynb notebook, change the necessary paths and run it. </p>

<h5>2.2 Evaluation </h5>
<p> In the test_model.pynb notebook. </p>

<h6>3. 3D WGAN </h6>
<h6> We were planning to train the pretrained model the <a  href= 'https://github.com/Hadrien-Cornier/E6040-super-resolution-project'> WGAN_citation </a> has on our 3D data, however we ran into some difficulties so we changed to the 2D model above.</h6>
