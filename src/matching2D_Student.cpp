#include <numeric>
#include "matching2D.hpp"

using namespace std;

#define WRITE_IMAGE

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string("Shi_Tomasi_") + to_string(img_num++) + string(".jpg"), visImage);
#endif
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    int block_size = 2;
    int aperture_size = 3;
    double k = 0.04;
    int threshold = 100;
    double overlap = 0;

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, block_size, aperture_size, k, cv::BORDER_DEFAULT);
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int i = 0; i < dst_norm_scaled.rows; i++){
        for (int j = 0; j < dst_norm_scaled.cols; j++){
            cv::KeyPoint cur_key_point(j, i, 2*aperture_size, -1, dst_norm_scaled.at<char>(i, j));
            if (cur_key_point.response >= threshold){
                bool to_add = true;
                for (auto kp: keypoints) {
                    if (cv::KeyPoint::overlap(cur_key_point, kp) > overlap){
                        if (cur_key_point.response>kp.response)
                            kp = cur_key_point;
                        to_add = false;
                        break;
                    }
                }
                if (to_add)
                    keypoints.push_back(cur_key_point);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis){
        std::string window_name = "Harris corner detector";
        cv::namedWindow(window_name, 6);
        cv::Mat vis_image = img.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, vis_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(window_name, vis_image);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string("HARRIS_") + to_string(img_num++) + string(".jpg"), vis_image);
#endif
    }
};

void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    int threshold = 100;
    double t = (double)cv::getTickCount();
    cv::FAST(img, keypoints, threshold);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST feature detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis){
        std::string window_name = "FAST feature detector";
        cv::namedWindow(window_name, 6);
        cv::Mat vis_image = img.clone();
        cv::drawKeypoints(img, keypoints, vis_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(window_name, vis_image);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string("FAST_") + to_string(img_num++) + string(".jpg"), vis_image);
#endif
    }
};

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis){
        std::string window_name = "BRISK feature detector";
        cv::namedWindow(window_name, 6);
        cv::Mat vis_image = img.clone();
        cv::drawKeypoints(img, keypoints, vis_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(window_name, vis_image);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string("BRISK_") + to_string(img_num++) + string(".jpg"), vis_image);
#endif
    }
};
