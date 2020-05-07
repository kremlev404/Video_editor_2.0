#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



class FaceDetector{
private:
	cv::String modelPath;
	cv::String configPath;
	cv::Size netSize;
	cv::Scalar mean;
	bool swapRB;
	double scale;
	float confidence_threshold;
	cv::dnn::Net net;
public:

	FaceDetector(cv::String modelPath, cv::String configPath,
		float confidence_threshold = 0.5, 
		int inputWidth = 300,
		int inputHeight = 300,
		double scale = 1.0,
		cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
		bool swapRB= false, 
		int backEnd = 0,
		int target = 0);
	std::vector<cv::Rect> detect(cv::Mat image);
};