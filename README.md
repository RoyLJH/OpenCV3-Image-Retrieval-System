# OpenCV3-Image-Retrieval-System
An image retrieval demo system based on OpenCV3.

This system includes 5 CBIR (content-based image retrieval) method.

1. Hash Perception (Color - based)

2. HSV Histogram (Color - based)

3. OTSU (Color - based , not typical)

4. GLCM (Texture - based)

5. Gloabl LBP (Texture - based)

This system is the course design for Searching Technology (2018 Fall, Advisor: Prof.Li Lian)

~~(I will not tell you I have done this in 24 hours)~~

If you want to know more about how this system is built or how each method performs , have a look at presention.pptx.


# Enviroment 
OpenCV 3.1.0

# How to use
1.Download the mirflickr25k dataset from : http://press.liacs.nl/mirflickr/mirdownload.html

2.Change the path of dataset to your downloaded directory in function **getPath(int)**

3.Use Training_xxx() method to save your image library fingerprints.

4.Use Retrieval_xxx() method to show the most matched result in your library to your chosen picture.

# Future work
1.Update more methods to do content-based image retrieval. (Including deep network)

2.Develop a user interaction framework (using QT maybe) , to guide the user choose picture and visualize the result.
