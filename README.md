# Neural Network Implementation

This project is a study of neural networks. You can find here, how I got introduced into this beautiful world of AI, and the amazing maths behind it. All the components that I created here, are fully reusable and functionals.

## Setting up the project

### opencv configuration

**Install required dependencies**

```
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libopencv-core-dev libopencv-highgui-dev libopencv-dev
```

**Clone the project from github**

```
git clone https://github.com/opencv/opencv.git
```

> *NOTE: opencv project is around 250MG of size.*

**Build opencv project**

Open a terminal in the directory of the cloned project and run the following commands to prepare the project dependencies and required stuff:

```
cd ~/opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

In a second step you need to build and install the opencv project:

```
make
sudo make install
```

### Project configurations

## Starting the project

## Contributions

All contributions are welcome, so, fill free to play with me and the AI on this project.
