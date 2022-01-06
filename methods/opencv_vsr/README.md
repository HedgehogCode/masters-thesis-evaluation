# OpenCV - VSR

Implementation of video super-resolution using OpenCV [`cv::superres::createSuperResolution_BTVL1()`](https://docs.opencv.org/4.5.4/d7/d0a/group__superres.html#ga73c184b0040c1afa7d6cdf0a9f32a8f8).

**Code**: https://github.com/opencv/opencv_contrib/

**Papers**:

- S. Farsiu, M. D. Robinson, M. Elad, and P. Milanfar, [“Fast and Robust Multiframe Super Resolution,”](https://ieeexplore.ieee.org/abstract/document/1331445) IEEE Trans. on Image Process., vol. 13, no. 10, pp. 1327–1344, Oct. 2004, doi: 10.1109/TIP.2004.834669.
- D. Mitzel, T. Pock, T. Schoenemann, and D. Cremers, [“Video Super Resolution Using Duality Based TV-L 1 Optical Flow,”](https://link.springer.com/chapter/10.1007/978-3-642-03798-6_44) in Pattern Recognition, vol. 5748, Springer Berlin Heidelberg, 2009, pp. 432–441. doi: 10.1007/978-3-642-03798-6_44.

## Usage

Dependencies

- [CMake](https://cmake.org/)
- [OpenCV](https://opencv.org/)

Creating the build files

```
$ mkdir build
$ cd build
$ cmake ..
```

Building the binary

```
$ cd build
$ make
```

Running the binary

```
$ cd build
$ ./SuperRes <path_to_input_video> <path_to_output_image>
```
