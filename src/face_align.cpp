#include "face_align.hpp"


std::vector<float> FaceAligner::landmarks_ref = { 0.31556875000000000, 0.4615741071428571,
							   0.68262291666666670, 0.4615741071428571,
							   0.50026249999999990, 0.6405053571428571,
							   0.34947187500000004, 0.8246919642857142,
							   0.65343645833333330, 0.8246919642857142 };

std::vector<cv::Point2f> coordsToPoints(std::vector<float>& pointsCoords) {
	std::vector<cv::Point2f> points;
	for (int i = 0; i < 10; i+= 2) {
		points.push_back(cv::Point2f(pointsCoords[i], pointsCoords[i + 1]));
	}
	return points;
}



cv::Mat FaceAligner::align(cv::Mat& face_crop, std::vector<float>& lands) {
	std::vector<cv::Point2f> lands_coords = coordsToPoints(lands);
	std::vector<cv::Point2f> lands_ref_coords = coordsToPoints(FaceAligner::landmarks_ref);

	for (int i = 0; i < 5; i++) {
		lands_coords[i].x *= face_crop.cols;
		lands_coords[i].y *= face_crop.rows;
		lands_ref_coords[i].x *= face_crop.cols;
		lands_ref_coords[i].y *= face_crop.rows;
	}

	cv::Mat output_image;
	cv::Mat trans_matrix = cv::estimateAffine2D(lands_coords, lands_ref_coords);

	cv::Size image_rectsize = face_crop.size();
	cv::warpAffine(face_crop, output_image, trans_matrix, face_crop.size(),1, cv::BORDER_REPLICATE);
	output_image = output_image(cv::Rect(0, 0, image_rectsize.width, image_rectsize.height));
	return output_image;
}

