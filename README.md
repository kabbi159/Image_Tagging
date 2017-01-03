# **Image_Tagging**
### **Implementations of Fast-Tag and Fast-Zero-Tag**
<br />

## **Code sources**
* We procured the codes for Deep Learning for Fast-Zero Tag from http://crcv.ucf.edu/projects/fastzeroshot/ and Fast Tag from www.cse.wustl.edu/~mchen/code/FastTag/fasttag.tar.gz.
* We implemented Rank-SVM for Fast-Tag ourselves.
* We also modified the above code for learning the kernelized mapping from image vector to principal
  direction in ranking SVM’s.
* We implemented the neural network(Figure 3) for fasttag in caffe[5].
* DATASET: IAPRTC-12
<br />

## **Acknowledgments**
We thank our instructor, Professor Piyush Rai for motivating us and guiding us throughout the project.
<br />

## **References**
[1] Minmin Chen (Amazon.com, Seattle, WA 98109), Alice Zheng (Microsoft Research, Redmond, WA
98052), Kilian Q. Weinberger (Washington University in St. Louis, St. Louis, MO 63130)Fast Image
Tagging 2013 <br />
[2] Yang Zhang, Boqing Gong, and Mubarak Shah, Center for Research in Computer Vision, University of
Central Florida, Orlando, FL 32816 Fast Zero-Shot Image Tagging 2016 <br />
[3] Guillaumin, M., Mensink, T., Verbeek, J., and Schmid, C. Tagprop: Discriminative metric learning in
nearest neighbor models for image auto-annotation. In Computer Vision, 2009 IEEE 12th International
Conference on, pp. 309–316. Ieee 2009 <br />
[4] Simonyan, K. and Zisserman, AVery Deep Convolutional Networks for Large-Scale Image Recognition.
CoRR ,abs/1409.1556 2014 <br />
[5] Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and
Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor, arXiv preprint arXiv:1408.5093, Caffe:
Convolutional Architecture for Fast Feature Embedding , 2014
