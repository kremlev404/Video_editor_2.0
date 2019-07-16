#pragma once
#include <string>

struct DetectedObject
{
	cv::Rect rect;
	int classId;
	float score;
};