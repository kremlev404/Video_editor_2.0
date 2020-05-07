#include "face_recognition.hpp"



FaceRecognizer::FaceRecognizer(cv::String modelPath, cv::String configPath, 
	int inputWidth,
	int inputHeight,
	double scale, 
	cv::Scalar mean,
	bool swapRB,
	int backEnd, 
	int target)
	: modelPath(modelPath), 
	configPath(configPath), 
	netSize(cv::Size(inputWidth, inputHeight)),
	scale(scale), mean(mean), swapRB(swapRB)
{
	net = cv::dnn::readNet(modelPath, configPath);
	net.setPreferableBackend(backEnd);
	net.setPreferableTarget(target);
}


std::vector<float> FaceRecognizer::predict(const cv::Mat& image) {
	std::vector<cv::Rect> detected_objects;

	cv::Mat resized_frame;
	cv::resize(image, resized_frame, netSize);

	cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);
	net.setInput(inputBlob);

	cv::Mat outBlob = net.forward();
	return std::vector<float>(outBlob.reshape(1, 1));
}

/*cosSimilarity = A*B/(|A|*|B|) */

float FaceRecognizer::cosSimilarity(std::vector<float>& first, std::vector<float>& second) {
	if (first.size() != second.size()) {
		throw "Vectors must have the same size";
	}
	size_t vec_size = first.size();
	float AB_numerator = 0;
	float AB_denominator = 0;
	float squares_sum_A = 0;
	float squares_sum_B = 0;

	for (size_t i = 0; i < vec_size; i++) {
		AB_numerator += first[i] * second[i];
		squares_sum_A += first[i] * first[i];
		squares_sum_B += second[i] * second[i];
	}
	AB_denominator = std::sqrt(squares_sum_A) * std::sqrt(squares_sum_B);
	return AB_numerator / AB_denominator;
}

/* @initial_person - contains person individual vectors(size: 256) from different photos
   @compare_person - with which we compare
*/
bool FaceRecognizer::compareManyFaces(std::vector<std::vector<float>>& initial_person, std::vector<float>& compare_person, float confidence_threshold, float votes_threshold) {
	int positive = 0; // positive votes
	if (initial_person.size() < 1) {
		throw "Person vector is empty";
	}
	for (std::vector<float>& photo : initial_person) {
		if (FaceRecognizer::cosSimilarity(photo, compare_person) > confidence_threshold) {
			positive++;
		}
	}
	if ((float)positive / initial_person.size() >= votes_threshold) {
		return true;
	}
	return false;
}
