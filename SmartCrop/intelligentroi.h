/* * SmartCrop - A tool for content aware croping of images
 * Copyright (C) 2024 Carl Philipp Klemm
 *
 * This file is part of SmartCrop.
 *
 * SmartCrop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SmartCrop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SmartCrop.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <opencv2/imgproc.hpp>

#include "yolo.h"

class InteligentRoi
{
private:
	int personId;
	static bool compPointPrio(const std::pair<cv::Point2i, int>& a, const std::pair<cv::Point2i, int>& b, const cv::Point2i& center);
	static void slideRectToPoint(cv::Rect& rect, const cv::Point2i& point);
	static cv::Rect maxRect(bool& incompleate, const cv::Size2i& imageSize, double targetAspectRatio, std::vector<std::pair<cv::Point2i, int>> mustInclude = {});

public:
	InteligentRoi(const Yolo& yolo);
	bool getCropRectangle(cv::Rect& out, const std::vector<Yolo::Detection>& detections, const cv::Size2i& imageSize, double targetAspectRatio);
};
