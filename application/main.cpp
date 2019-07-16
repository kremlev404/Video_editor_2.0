#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "face_detector.hpp"

using namespace cv;
using namespace std;

static const char* keys =
"{ i  image         | <none> | image to process        }"
"{ model_path       | <none> | width for image resize  }"
"{ config_path        | <none> | height for image resize }"
"{ q ? help usage   | <none> | print help message      }";



int main(int argc, char** argv) {

	CommandLineParser parser(argc, argv, keys);


	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	String image_path = parser.get<String>("image");
	String model_path = parser.get<String>("model_path");\
	String config_path = parser.get<String>("config_path");
	



	Mat image = imread(image_path);


	FaceDetector detector(model_path, config_path, 300, 300, false, 0.6);
	namedWindow("FaceDetect");

	vector<DetectedObject> detected_faces = detector.Detect(image);

	for (DetectedObject face : detected_faces) {
		rectangle(image, face.rect, Scalar(0, 255, 0), 3);
	}
	imshow("FaceDetect", image);

	waitKey();

	

}