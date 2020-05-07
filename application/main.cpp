#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio/videoio_c.h>
#include <stdlib.h>
#include <stdio.h>

#include "face_detector.hpp"
#include "landmarks_detector.hpp"
#include "face_align.hpp"
#include "face_recognition.hpp"
#include "video_assistance.hpp"
#include "video_gui.hpp"

using namespace cv;
using namespace std;

static const char* keys =
"{ i  images_path         | <none> | path to images }"
"{ models_path      | <none>	| path to models  }"
"{ v video_path      | <none>	| path to models  }"

"{ n numbers        | <none> | number of images to process }"
"{ o outpath       | <none> | path to save clips }"
"{ q ? help usage   | <none> | print help message      }";






int main(int argc, char** argv) {

	CommandLineParser parser(argc, argv, keys);
	
	if (!parser.check())
	{
		parser.printErrors();
		throw "Parse error";
		return 0;
	}

	String detection_model_path = parser.get<String>("models_path") + "/face-detection-retail-0004.xml";
	String detection_config_path = parser.get<String>("models_path") + "/face-detection-retail-0004.bin";
	String landmarks_model_path = parser.get<String>("models_path") + "/landmarks-regression-retail-0009.xml";
	String landmarks_config_path = parser.get<String>("models_path") + "/landmarks-regression-retail-0009.bin";
	String recognizer_config_path = parser.get<String>("models_path") + "/face-reidentification-retail-0095.bin";
	String recognizer_model_path = parser.get<String>("models_path") + "/face-reidentification-retail-0095.xml";

	String video_path = parser.get<String>("video_path");

	String out_path = parser.get<String>("outpath"); // path where we will save cropped video clips;

	String images_path = parser.get<String>("images_path"); // path where we will save cropped video clips;



	int images_number = parser.get<int>("numbers");

	vector<Mat> images;
	for (int i = 1; i < images_number + 1; i++) {
		images.push_back(imread(images_path + "/" + to_string(i) + ".jpg"));
		if (images[i - 1].empty()) {
			std::cout << "Image reading error" << std::endl;
		}

	}


	/* Detectors */
	FaceDetector face_detector(detection_model_path, detection_config_path);
	LandmarkDetector landmarks_detector(landmarks_model_path, landmarks_config_path);
	FaceRecognizer recognizer(recognizer_model_path, recognizer_config_path);
	FaceAligner aligner;


	VideoAssistant assistant = VideoAssistant(video_path, images, face_detector, landmarks_detector, recognizer);
	
	assistant.saveFragments(out_path);


	
	return 0;
}