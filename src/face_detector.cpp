#include "face_detector.hpp"




FaceDetector::FaceDetector(cv::String _modelPath, cv::String _configPath, int _inputWidth, int _inputHeight, bool _swapRB,
	float confidence_threshold, double _scale, cv::Scalar _mean, int _backEnd, int _target) :
	modelPath(_modelPath), configPath(_configPath), netSize(cv::Size(_inputWidth, _inputHeight)), confidence_threshold(confidence_threshold), scale(_scale), mean(_mean), swapRB(_swapRB) {


	net = cv::dnn::readNet(modelPath, configPath);
	net.setPreferableBackend(_backEnd);
	net.setPreferableTarget(_target);
}


std::vector<DetectedObject> FaceDetector::Detect(cv::Mat image) {
	std::vector<DetectedObject> detected_objects;

	cv::Mat resized_frame;
	cv::resize(image, resized_frame, netSize);
	cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);

	net.setInput(inputBlob);
	cv::Mat detection = net.forward();
	cv::Mat detection_as_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


	for (int i = 0; i < detection_as_mat.rows; i++)
	{
		float cur_confidence = detection_as_mat.at<float>(i, 2);
		int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
		int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * image.cols);
		int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * image.rows);
		int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * image.cols);
		int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * image.rows);
		cv::Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

		if (cur_confidence < confidence_threshold)
			continue;

		if (cur_rect.empty())
			continue;

		cur_rect = cur_rect & cv::Rect(cv::Point(), image.size());
		DetectedObject cur_obj = { cur_rect, cur_class_id, cur_confidence };
		detected_objects.push_back(cur_obj);
	}
	return detected_objects;
}