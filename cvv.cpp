//============================================================================
// Name        : Lab1.cpp
// Author      : Menicatti & Sorial
// Version     :
// Copyright   : Your copyright notice
// Description : COVIS Lab1 - Tracking object from first frame
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	VideoCapture cap("/home/msorial/workspace/Lab1/video1.mp4");

	Mat video;
	Mat video_grey;
	Mat frame1;
	Mat keypoints_video;
	Mat keypoints_frame1;
	Mat descriptors_video;
	Mat descriptors_frame1;
	Mat video_matches;
	vector<KeyPoint> keypointsVector_video, keypointsVector_frame1;
	vector<DMatch> matches;
	SurfFeatureDetector detector(2000);
	FlannBasedMatcher matcher;

// FIRST FRAME

	cap.read(frame1);						//reading first frame from video file
	cvtColor(frame1, frame1, CV_RGB2GRAY);  //converting the frame1 into grey scale (keeping the same name)

	//Lime key points extraction to the the square of the ball
	Mat mask(frame1.size(), CV_8UC1, Scalar::all(0));
	Rect ROI(24, 44, 172, 166);
	mask(ROI).setTo(Scalar::all(1));

	// Detecting key points and calculating descriptors of the first frame
	detector.detect(frame1, keypointsVector_frame1, mask);
	detector.compute(frame1, keypointsVector_frame1, descriptors_frame1);
	drawKeypoints(frame1, keypointsVector_frame1, keypoints_frame1,
			Scalar::all(-1), DrawMatchesFlags::DEFAULT);

// VIDEO

	cout << "Press ESC key to close windows." << endl;

	while (1) {
		bool bSuccess = cap.read(video); 		//read a new frame from video
		if (!bSuccess){ 						//if not success, break loop
			cout << "The video is over or it has not been found." << endl;
			break;
		}

		cvtColor(video, video_grey, CV_RGB2GRAY); //convert the video into grey scale

		// Detecting key points and calculating descriptors of the video

		detector.detect(video_grey, keypointsVector_video);
		detector.compute(video_grey, keypointsVector_video, descriptors_video);
		drawKeypoints(video_grey, keypointsVector_video, keypoints_video,
				Scalar::all(-1), DrawMatchesFlags::DEFAULT);

		// Matching key points of the video with key points of the first frame

		matcher.match(descriptors_frame1, descriptors_video, matches);

		vector<DMatch> good_matches;
		for (unsigned int i = 0; i < matches.size(); i++) {
			if (matches[i].distance < 0.4) {
				good_matches.push_back(matches[i]);
			}
		}

		Mat img_matches;
		drawMatches(frame1, keypointsVector_frame1, video_grey,
				keypointsVector_video, good_matches, img_matches,
				Scalar::all(-1), Scalar::all(-1), vector<char>(),
				DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object
		vector<Point2f> obj;
		vector<Point2f> scene;

		for (unsigned int i = 0; i < good_matches.size(); i++) {
			//-- Get the keypoints from the good matches
			obj.push_back(keypointsVector_frame1[good_matches[i].queryIdx].pt);
			scene.push_back(keypointsVector_video[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )

		vector<Point2f> obj_corners(4);

		obj_corners[0] = cvPoint( 24,  44);
		obj_corners[1] = cvPoint(196,  44);
		obj_corners[2] = cvPoint(196, 210);
		obj_corners[3] = cvPoint( 24, 210);

		vector<Point2f> scene_corners(4);
		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(frame1.cols, 0),
				scene_corners[1] + Point2f(frame1.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners[1] + Point2f(frame1.cols, 0),
				scene_corners[2] + Point2f(frame1.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners[2] + Point2f(frame1.cols, 0),
				scene_corners[3] + Point2f(frame1.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners[3] + Point2f(frame1.cols, 0),
				scene_corners[0] + Point2f(frame1.cols, 0), Scalar(0, 255, 0),
				4);

//-- Show detected matches

		imshow("Good Matches & Object detection", img_matches);

// Exit loop by pressing ESC

		char c = cvWaitKey(33);
		if (c == 27)
			break;

	}

	waitKey(0);
	return 0;

}
