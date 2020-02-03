# Traffic-Sign-Recognition

With the development of intelligent transportation, traffic sign recognition system has become an important task for road maintenance, traffic accident reduction and automatic driving system.Although previous studies on the detection and classification of traffic signs have achieved gratifying results, there are few studies on the detection of traffic signs in the real world.In addition, compared with other countries' traffic signs, China's traffic signs have their own unique characteristics. In reality, the road traffic environment is complex and changeable, and the practical application requires it to ensure high accuracy and real-time performance.

combined with the research background that the resolution of traffic signs in real scenes is relatively small and the detection speed of small targets is guaranteed by the YOLOv3 algorithm, the YOLOv3 algorithm is finally selected to complete the identification of traffic signs.And compared with other algorithms.Among them, darknet-53 extracts the characteristics of the input image. The feature extraction network is a Residual network formed by Residual unit, which can effectively control the spread of gradient and prevent gradient explosion or disappearance, etc., which is not conducive to training.Compared with other algorithms in the YOLO series, the accuracy is significantly improved.After that, the semantic fusion method similar to Fpn feature pyramid was adopted in layer 75-105 to fuse the high-level features of low-resolution and high-semantic information with the low-level features of high-resolution and low-semantic information.To improve the detection accuracy of traffic signs that take up a small proportion in the image.

Our final effect is shown below：
![image](https://github.com/YufeiHU-fr/Traffic-Sign-Recognition/blob/master/yolov3/detection/00023.jpg)

For video detection, we get the following results：
![image]()


