# Hand Detection and Distance Estimation
Uses Tensorflow Object Detection API to detect and track hands in real-time, as well as estimate their distance from the camera

![hand_detection](https://user-images.githubusercontent.com/25287427/42243870-bb1112a0-7ee0-11e8-9054-9bfe80a950a7.gif)

## Getting Started

### Prerequisites
This projects requires that you have [Anaconda](https://www.anaconda.com/download/#linux) installed to get the development enviroment installed

### Installation
Included is a .yml file that installs the necessary compoments for this to run.
Run the following bash file to setup the enviroment and activate it:
``` bash
bash setup_enviroment.sh
```
To run the hand detection python script run:
``` bash
python hand_detection
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Victor Dibia - [Real-time Hand Detection](https://github.com/victordibia/handtracking)
* Adrian Rosebrock - [Find distance from a camera to object/marker using Python and OpenCV](https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/)
* Edje Electronics - [Training Custom Object Detector Using Tensorflow](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
* [EgoHands dataset](http://vision.soic.indiana.edu/projects/egohands/)
* [VIVA Hand Detection dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/)
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
