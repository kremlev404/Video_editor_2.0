#include "landmarks_detector.hpp"



LandmarkDetector::LandmarkDetector(cv::String _modelPath, cv::String _configPath, 
	int _inputWidth,
	int _inputHeight,
	double _scale,
	cv::Scalar _mean,
	bool _swapRB, 
	int _backEnd,
	int _target) 
	:modelPath(_modelPath),
	configPath(_configPath),
	netSize(cv::Size(_inputWidth, _inputHeight)),
	scale(_scale), mean(_mean), swapRB(_swapRB)
{


	net = cv::dnn::readNet(modelPath, configPath);
	net.setPreferableBackend(_backEnd);
	net.setPreferableTarget(_target);
}


std::vector<float> LandmarkDetector::detect(cv::Mat image) {
	std::vector<float> detected_landmarks;

	cv::Mat resized_frame;
	cv::resize(image, resized_frame, netSize);

	cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);
	net.setInput(inputBlob);


	cv::Mat outBlob = net.forward();
	for (int i = 0; i < 10; i++) { // landmark regression. Output is 10 coordinates x0, y0, x1, y1, ..., x4, y4
		detected_landmarks.push_back(outBlob.reshape(1, 1).at<float>(0, i));
	}



	return detected_landmarks;
}

