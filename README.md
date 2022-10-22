# ZSPD-Deep-learning-projekt
# Heart MRI Superresolution

<p> Members of the team: 
<ul>
  <li>Fris Domonkos - WPOU2Z</li>
  <li>Homolya Panni - ARQ7TQ</li>
  <li>Turcsik Zsófia - R7UJDO</li>
</ul>
</p>
Making high resolution MRI images takes quite a long time and the movement of the patient makes it difficult to make high quality images, but from the low resolution pictures it is harder to gather information about the subject.

In this projeck our objective is to convert low resolution heart MRIs to high resolution images with deep learning techniques. Our initial data consists of 4 dimensional high resolution images of shape $(216,256,10,t)$, where $t$ denotes time and differs troughout the 100 patients, but it's usually around 30. We also have some ground truth files, which helps with the segmentation of the pictures and includes the position of the heart on the MRIs.

With the __codename__ codes we reshaped the data and saved the 4D pictures to $10t$ different 2D images, then saved them to training, validation and test folders, still separating the data by patients. We also saved the values of the bounding boxes for each picture in a text file for each folder.

In the data_preparation code we prepared for converting the pictures to lower resolution images and prepared for segmentation.
