#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


class FaceAligner{
public:
	static std::vector<float> landmarks_ref;
	static cv::Scalar mean(const cv::Mat& matrix);
	static float std(const cv::Mat matrix, const cv::Scalar& mean);
	static cv::Mat align(cv::Mat& image, std::vector<float>& landmarks);
};

