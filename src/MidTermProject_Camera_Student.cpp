/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>
#include <cmath>
#include <limits>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

template<class T>
class CircleBuffer : public list<T>{
public:
    class iterator{
    private:
        typename list<T>::iterator it_;
    public:
        iterator(typename list<T>::iterator i){it_ = i;};
        iterator& operator-(int v){
            for (int i = 0; i < v; i++ )
                it_--;
            return *this;
        };
        bool operator!=(const iterator& it){
            return it_ != it.it_;
        };
        T& operator*(){return *it_;};
        typename list<T>::iterator operator->(){return it_;};
    };
    void push_back(const T& x){
        if (list<T>::size() == data_buffer_size_){
            list<T>::erase(list<T>::begin());
        }
        list<T>::push_back(x);
    };
    CircleBuffer(int dataBufferSize):data_buffer_size_(dataBufferSize){};
    iterator begin(){return iterator(list<T>::begin());};
    iterator end(){return iterator(list<T>::end());};
    private:
    int data_buffer_size_;
};

bool ReadCommandLine(int argc, const char* argv[], string& detectorType, string& descriptorType, string& selectorType, string& matcherType, bool& bVis){
    int i = 1;
    string command_line_help = "-det detectorType -des descriptorType -sel selectorType -mat matcherType -vis";
    if ( (argc == 2) && (std::string(argv[1]) == std::string("-h") ) ){
        std::cout << command_line_help << std::endl;
        return -1;
    }

    while(i < argc){
        std::string us_in = string(argv[i++]);
        if  (us_in == "-det")
            detectorType = argv[i++];
        else if (us_in == "-des")
            descriptorType = argv[i++];
        else if (us_in == "-sel")
            selectorType = argv[i++];
        else if (us_in == "-mat")
            matcherType = argv[i++];
        else if (us_in == "-vis")
            bVis = strcmp(argv[i++], "0")==0? false: true;
        else{
            std::cout << "Invalid command line" << std::endl;
            std::cout << command_line_help << std::endl;
            return -1;
        }
    }
    return 0;
};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2; // no. of images which are held in memory (ring buffer) at the same time
    CircleBuffer<DataFrame> dataBuffer(dataBufferSize); // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    string detectorType("SHITOMASI");
    string descriptorType("BRISK");
    string matcherType ("MAT_BF");
    string selectorType("SEL_KNN");

    // detectorType SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    // descriptorType BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    // selectorType SEL_NN, SEL_KNN
    if (ReadCommandLine(argc, argv, detectorType, descriptorType, selectorType, matcherType, bVis))
        return -1;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// Done MP.1 -> replace the following code with ring buffer of size dataBufferSize
        // Done in class CircleBuffer line 23

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// Done MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
        else if (detectorType.compare("HARRIS") == 0)
            detKeypointsHarris(keypoints, imgGray, bVis);
        else
            detKeypointsModern(keypoints, imgGray, detectorType, bVis);

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// Done MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            std::vector<cv::KeyPoint> keypoints_filtered;
            copy_if(keypoints.begin(),
            keypoints.end(),
            std::back_inserter(keypoints_filtered),
            [&](const cv::KeyPoint& el){
                return
                (el.pt.x >= vehicleRect.x)&&
                (el.pt.x <= (vehicleRect.x+vehicleRect.width)) &&
                (el.pt.y >= vehicleRect.y) &&
                (el.pt.y <= (vehicleRect.y + vehicleRect.height));});
            keypoints = std::move(keypoints_filtered);
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// Done MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            string descriptorTypeBH = "DES_BINARY"; // DES_BINARY, DES_HOG
            if (descriptorType.compare("SIFT") == 0)
                descriptorTypeBH = "DES_HOG";

            //// STUDENT ASSIGNMENT
            //// DONE MP.5 -> add FLANN matching in file matching2D.cpp
            //// DONE MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorTypeBH, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}
