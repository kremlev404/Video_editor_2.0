#include "video_assistance.hpp"
#include <string>


VideoAssistant::VideoAssistant(cv::String videoPath, std::vector<cv::Mat>& person_images,
	FaceDetector& fdetector, LandmarkDetector& ldetector, FaceRecognizer& recognizer, int frames_skip, float max_interrupt) : 
	videoPath(videoPath), fdetector(fdetector), ldetector(ldetector), recognizer(recognizer), frames_skip(frames_skip), max_interrupt(max_interrupt) {
	if (frames_skip < 1) throw "frames_skip variable must be greater than 1 or equal 1";
	for (cv::Mat& image : person_images) {
		std::vector<cv::Rect> face_arr = fdetector.detect(image);
		if (face_arr.size() != 1) {
			throw "Can't definitely recognize a person (The image must contain strictly one face.)";
		}
		cv::Rect face_rect = face_arr[0];
		std::vector<float> face_landmarks = ldetector.detect(image(face_rect));
		cv::Mat transformedFace = FaceAligner::align(image(face_rect), face_landmarks);
		face_representations.push_back(recognizer.predict(transformedFace));
	}
	cv::VideoCapture video(videoPath);
	frame_size = cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT));
};


std::vector<int> VideoAssistant::getFrames() {
	cv::Mat frame;
	cv::VideoCapture capture(videoPath);
	std::vector<int> matched_frames;
	int frame_counter = 1;
	/* loading bar*/

	std::string frameState = "\r" + std::to_string(frame_counter) + " / " + std::to_string(capture.get(cv::CAP_PROP_FRAME_COUNT));
	std::cout << frameState;


	while (frame_counter < capture.get(cv::CAP_PROP_FRAME_COUNT)) {
		capture >> frame;
		if (frame.empty()) {
			frame_counter++;
			continue;
		}
		std::vector<cv::Rect> faces = fdetector.detect(frame);
		for (cv::Rect& face : faces) {
			std::vector<float> landmarks = ldetector.detect(frame(face));
			cv::Mat transformedFace = FaceAligner::align(frame(face), landmarks);
			std::vector<float> face_repr = recognizer.predict(transformedFace);
			bool personFound = FaceRecognizer::compareManyFaces(face_representations, face_repr);
			if (personFound) {
				matched_frames.push_back(frame_counter);
				break;
			}
		}
		frame_counter += frames_skip;
		/* Loading bar */
		if (frame_counter % 20 == 0) {
			frameState = "\r" + std::to_string(frame_counter) + " / " + std::to_string(capture.get(cv::CAP_PROP_FRAME_COUNT));
			std::cout << frameState;

		}

	}
	return matched_frames;
}

std::vector<std::pair<int, int>> VideoAssistant::getFramesIntervals(std::vector<int>& frames) {
	std::vector<std::pair<int, int>> intervals;
	cv::VideoCapture capture(videoPath);
	float FPS = capture.get(cv::CAP_PROP_FPS);
	int max_interrupt_frames = max_interrupt * FPS; // if person not attend in this amount of frames, we consider this as one segment
	int startInterval = frames[0];
	int endInterval = frames[0];
	for (int i = 1; i < frames.size(); i++) {
		if (frames[i] - frames[i - 1] < max_interrupt_frames) { // if difference between frames number more then max_interrupt, then it's the new interval
			endInterval = frames[i];
		}else {
			intervals.push_back(std::make_pair(startInterval, endInterval));
			startInterval = frames[i];
			endInterval = frames[i];
		}
	}
	intervals.push_back(std::make_pair(startInterval, endInterval));
	return intervals;
}

void VideoAssistant::saveFragments(cv::String path) {
	std::vector<int> frames = getFrames();
	cv::VideoCapture capture(videoPath);

	std::vector<std::pair<int, int>> intervals = getFramesIntervals(frames);
	float FPS = capture.get(cv::CAP_PROP_FPS);
	int frame_counter = 1;
	for (std::pair<int, int> interval : intervals) {
		cv::String videoname = path + "/" + std::to_string(interval.first) + "-" + std::to_string(interval.second) + ".avi";
		cv::VideoWriter out(videoname, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), FPS, frame_size);
		cv::Mat current_frame;
		while (frame_counter <= interval.second) {
			capture >> current_frame;
			if (current_frame.empty() || frame_counter < interval.first) {
				frame_counter++;
				continue;
			}
			out.write(current_frame);
			frame_counter++;
		}
		out.release();
	}
	capture.release();
}

cv::String VideoAssistant::getPath() {
	return cv::String(videoPath);
}