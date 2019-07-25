#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


#include "face_detector.hpp"
#include "face_recognition.hpp"
#include "landmarks_detector.hpp"
#include "face_align.hpp"
#include "face_recognition.hpp"

class VideoAssistant {
private:
	cv::String videoPath;
	FaceDetector& fdetector;
	LandmarkDetector& ldetector;
	FaceRecognizer& recognizer;
	cv::Size frame_size;
	std::vector < std::vector<float>> face_representations; // vector representation of person face which we need to recognize
	float max_interrupt; // in seconds
	int frames_skip;

public:
	VideoAssistant(cv::String videoPath,
		std::vector<cv::Mat>& person_images,
		FaceDetector& fdetector,
		LandmarkDetector& ldetector, 
		FaceRecognizer& recognizer, 
		int frames_skip = 1, 
		float max_interrupt = 1.0);

	std::vector<int> getFrames();
	std::vector<std::pair<int, int>> getFramesIntervals(std::vector<int>& frames);
	void saveFragments(cv::String path);
	cv::String getPath();
};


