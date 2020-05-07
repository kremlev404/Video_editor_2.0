#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



class FaceRecognizer {
private:
	cv::String modelPath;
	cv::String configPath;
	cv::Size netSize;
	cv::Scalar mean;
	bool swapRB;
	double scale;
	cv::dnn::Net net;

public:

	FaceRecognizer(cv::String modelPath, cv::String configPath, 
		int inputWidth = 128,
		int inputHeight = 128,
		double scale = 1.0, 
		cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
		bool swapRB = false, 
		int backEnd = 0, 
		int target = 0);
	std::vector<float> predict(const cv::Mat& image);
	static float cosSimilarity(std::vector<float>& first, std::vector<float>& second);
	static bool compareManyFaces(std::vector<std::vector<float>>& initial_person, std::vector<float>& compare_person,
		float confidence_threshold = 0.34, float votes_threshold = 0.5); // true - it's the same person
};