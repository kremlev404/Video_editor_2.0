#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedObject.hpp"




class Detector
{

public:
	virtual std::vector<DetectedObject> Detect(cv::Mat image) = 0 {}
};


class FaceDetector : public Detector {
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

	FaceDetector(cv::String _modelPath, cv::String _configPath, int _inputWidth, int _inputHeight, bool _swapRB, float confidence_threshold,
		double _scale = 1.0, cv::Scalar _mean = cv::Scalar(0, 0, 0, 0), int _backEnd = 0, int _target = 0);
	std::vector<DetectedObject> Detect(cv::Mat image);
};