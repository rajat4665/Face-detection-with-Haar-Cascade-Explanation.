# Face-detection-with-Haar-Cascade-Explanation.
In this repository I am gonna show you how ine can detect face from an image using Haar cascade and understang its working.
<h3>A pipeline of Haar Cascade Face Detection</h3>
<img class="alignnone size-full wp-image-123" src="https://getpython.files.wordpress.com/2019/07/screenshot-from-2019-07-09-00-49-51.png" alt="Screenshot from 2019-07-09 00-49-51" width="749" height="262" />
<h3></h3>
<h3>How to run this code:</h3>
<ul>
	<li>download this code from my GitHub</li>
	<li>clone this repository</li>
	<li>open it into Jupyter notebook</li>
	<li>Now run its cells one by one</li>
</ul>
<h3>How to install jupyter notebook in Ubuntu:</h3>
open your terminal and paste these commands one by one.
<pre class="code-pre command"><code>sudo apt install python3-pip python3-dev</code></pre>
<pre class="code-pre command"><code>pip install jupyter</code></pre>
<h3>How to install jupyter notebook in windows:</h3>
Follow this <a href="https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/install.html">Link for windows</a>
<h3></h3>
<h3>Introduction:</h3>
Hello readers how are you doing, I hope you guys doing great.  As I have already discussed one Face detection algorithm using MTCNN which is widely used nowadays. But Today I am gonna show you  Face detection using  Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones. It was launched in 2001.
<h3>Explanation:</h3>
It is a machine learning based approach where a Cascade function is trained to solve a binary problem to detect that image contains a face or not face. It is trained on a lot of having faces (positive) and without faces (negative) images.

First, It detects Haar features from an image, Haar features from an image. Basically, Haar features are the gradient measurements from a rectangular region. Haar features detect Edges, lines, and Rectangular pattern in order to detect face from an image. In the case of face detection lines and rectangle specially used. These features effectively identify the extreme area of the bright and dark region on a human face.

<img class=" size-full wp-image-122 aligncenter" src="https://getpython.files.wordpress.com/2019/07/haar_features.jpg" alt="haar_features" width="320" height="271" />

<strong>Working of haar filters:</strong>

As you can see the haar features from the above image, select any one haar feature from them. I hope you choose one of them, Have you guys wondered why it is consist of white and black color rectangles. These are a representation of mathematical filter in the frequency domain , when these filter applied to an image it extracts features. Here white rectangle represents that it allows image data from the white region and block data from the black rectangle for more information please study about image processing in the frequency domain.

Before going further remember we will provide these features on a grayscale image because for face detection grayscale image is  perfect (fast process due to less information)

<strong>Step 1: </strong>Vertical Line Features Detection

In this step, our filter applied on a gray image and generate a new image which is called a featured image. It detects faces from the image and removes an unnecessary background. It removes background because it reduces information (image information is directly proportional to the speed of processing)

<strong>Step 2: </strong>Rectangle Feature Detection

In this step, It applies the filter to discards unnecessary information from a line filtered image like we don't need chest, Shoulder, neck for face detection so this rectangle filter marks them unnecessary and discards them. Finally, we got a lightweight image which is easy to process.
<h3><strong>Source Code:</strong></h3>
<pre># import required libraries for this section
import numpy as np
import matplotlib.pyplot as plt
import cv2</pre>
<pre># load in color image for face detection
image = cv2.imread('group4.jpg')

# convert to RBG
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.imshow(image)

</pre>
<pre># convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

plt.figure(figsize=(20,10))
plt.imshow(gray, cmap='gray')</pre>
<pre># load in cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(gray)       

</pre>
<pre>print("Total faces :", len(faces))

img_with_detections = np.copy(image)   

for (x,y,w,h) in faces:
    cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(0,255,0),5)  

# display the result
plt.figure(figsize=(20,10))
plt.imshow(img_with_detections)</pre>
<h3></h3>
<h3>Input Image :</h3>
<img class="alignnone size-full wp-image-124" src="https://getpython.files.wordpress.com/2019/07/screenshot-from-2019-07-09-01-24-23.png" alt="Screenshot from 2019-07-09 01-24-23" width="576" height="347" />
<h3>Output :</h3>
<img class="alignnone size-full wp-image-125" src="https://getpython.files.wordpress.com/2019/07/screenshot-from-2019-07-09-01-24-39.png" alt="Screenshot from 2019-07-09 01-24-39" width="581" height="383" />
<h3></h3>
<h3>Conclusion:</h3>
It is undeniable fact that Haar  Cascade has immense contribution in the world of face detection but due to less accuracy nowadays people prefer Deep learning model such ass MTCNN, Vgg16 over Haar Cascade to achieve much better accuracy and fast processing.

 

References:

<a href="https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html">Face Detection using Haar Cascades</a>

<a href="https://unsplash.com/@naassomz1" target="_blank" rel="noopener">Image_credit</a>
