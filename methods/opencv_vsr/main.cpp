#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/superres.hpp>
using namespace cv;
using namespace std;

// BTVL1 parameters
const int BTVL1_ITERATIONS = 100;     // default: 100
const double BTVL1_LAMBDA = 0.03;     // default: 0.03
const double BTVL1_TAU = 1.3;         // default: 1.3
const double BTVL1_ALPHA = 0.7;       // default: 0.7
const int BTVL1_BTV_KERNEL_SIZE = 7;  // default: 7
const int BTVL1_BLUR_KERNEL_SIZE = 5; // default: 5
const double BTVL1_BLUR_SIGMA = 0.0;  // default: 0.0

// DUAL-TVL1 parameters
const double DUALTVL1_EPSILON = 0.01;    // default: 0.01
const int DUALTVL1_ITERATIONS = 10;      // default: 10
const double DUALTVL1_LAMBDA = 0.15;     // default: 0.15
const int DUALTVL1_SCALES_NUMBER = 5;    // default: 5
const double DUALTVL1_TAU = 0.25;        // default: 0.25
const double DUALTVL1_THETA = 0.3;       // default: 0.3
const int DUALTVL1_USE_INITIAL_FLOW = 0; // default: 0
const int DUALTVL1_WARPINGS_NUMBER = 5;  // default: 5

// Farneback parameters
const double FARNEBACK_PYR_SCALE = 0.5;  // default: 0.5
const int FARNEBACK_LEVELS_NUMBER = 5;   // default: 5
const int FARNEBACK_WINDOW_SIZE = 13;    // default: 13
const int FARNEBACK_ITERATIONS = 10;     // default: 10
const int FARNEBACK_POLY_N = 5;          // default: 5
const double FARNEBACK_POLY_SIGMA = 1.1; // default: 1.1
const int FARNEBACK_FLAGS = 0;           // default: 0

Mat computeBTVL1SuperRes(
    int scale,
    int numFrames,
    Ptr<superres::FrameSource> frameSource,
    Ptr<superres::DenseOpticalFlowExt> opticalFlow)
{
    int targetIndex = numFrames / 2;
    int temporalRadius = targetIndex;

    Ptr<superres::SuperResolution> superRes = superres::createSuperResolution_BTVL1();

    // Set algorithm parameters
    superRes->setIterations(BTVL1_ITERATIONS);
    superRes->setLambda(BTVL1_LAMBDA);
    superRes->setTau(BTVL1_TAU);
    superRes->setAlpha(BTVL1_ALPHA);
    superRes->setKernelSize(BTVL1_BTV_KERNEL_SIZE);
    superRes->setBlurKernelSize(BTVL1_BLUR_KERNEL_SIZE);
    superRes->setBlurSigma(BTVL1_BLUR_SIGMA);

    // Set input paramters
    superRes->setScale(scale);
    superRes->setTemporalAreaRadius(temporalRadius);
    superRes->setOpticalFlow(opticalFlow);
    superRes->setInput(frameSource);

    // Compute the result
    Mat result;
    for (int i = 0; i <= targetIndex; i++)
    {
        superRes->nextFrame(result);
    }
    return result;
}

Ptr<superres::DenseOpticalFlowExt> getDualTVL1OpticalFlow()
{
    Ptr<superres::DualTVL1OpticalFlow> algo = superres::createOptFlow_DualTVL1();
    algo->setEpsilon(DUALTVL1_EPSILON);
    algo->setIterations(DUALTVL1_ITERATIONS);
    algo->setLambda(DUALTVL1_LAMBDA);
    algo->setScalesNumber(DUALTVL1_SCALES_NUMBER);
    algo->setTau(DUALTVL1_TAU);
    algo->setTheta(DUALTVL1_THETA);
    algo->setUseInitialFlow(DUALTVL1_USE_INITIAL_FLOW);
    algo->setWarpingsNumber(DUALTVL1_WARPINGS_NUMBER);
    return algo;
}

Ptr<superres::DenseOpticalFlowExt> getFarnebackOpticalFlow()
{
    Ptr<superres::FarnebackOpticalFlow> algo = superres::createOptFlow_Farneback();
    algo->setPyrScale(FARNEBACK_PYR_SCALE);
    algo->setLevelsNumber(FARNEBACK_LEVELS_NUMBER);
    algo->setWindowSize(FARNEBACK_WINDOW_SIZE);
    algo->setIterations(FARNEBACK_ITERATIONS);
    algo->setPolyN(FARNEBACK_POLY_N);
    algo->setPolySigma(FARNEBACK_POLY_SIGMA);
    algo->setFlags(FARNEBACK_FLAGS);
    return algo;
}

void waitForever()
{
    while (true)
    {
        waitKey(0);
    }
}

int countFrames(const string videoPath)
{
    Ptr<superres::FrameSource> frameSource = superres::createFrameSource_Video(videoPath);
    Mat result;
    for (int i = 0; i < std::numeric_limits<int>::max(); i++)
    {
        frameSource->nextFrame(result);
        if (result.total() <= 0)
        {
            return i;
        }
    }
    throw std::runtime_error("Video has too many frames.");
}

int main(int argc, char **argv)
{
    // Arguments
    const string videoPath = argv[1];
    const string resultPath = argv[2];
    const int scale = 4;

    // Count the number of frames in the image
    const int numFrames = countFrames(videoPath);

    // Open the video
    Ptr<superres::FrameSource> frameSource = superres::createFrameSource_Video(videoPath);

    // Define the optical flow
    Ptr<superres::DenseOpticalFlowExt> opticalFlow = getDualTVL1OpticalFlow();
    // Ptr<superres::DenseOpticalFlowExt> opticalFlow = getFarnebackOpticalFlow();

    // Run super resolution
    Mat result = computeBTVL1SuperRes(scale, numFrames, frameSource, opticalFlow);
    imwrite(resultPath, result);

    // Show the result
    // namedWindow("SR Image", WINDOW_NORMAL);
    // resizeWindow("SR Image", 1000, 1000);
    // imshow("SR Image", result);
    // waitForever();

    return 0;
}
