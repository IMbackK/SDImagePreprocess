//
// SmartCrop - A tool for content aware croping of images
// Copyright (C) 2024 Carl Philipp Klemm
//
// This file is part of SmartCrop.
//
// SmartCrop is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SmartCrop is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with SmartCrop.  If not, see <http://www.gnu.org/licenses/>.
//

#include "utils.h"

#include <filesystem>
#include <vector>
#include <opencv2/imgproc.hpp>

bool isImagePath(const std::filesystem::path& path)
{
	return std::filesystem::is_regular_file(path) && (path.extension() == ".png" || path.extension() == ".jpg" || path.extension() == ".jpeg");
}

void getImageFiles(const std::filesystem::path& path, std::vector<std::filesystem::path>& paths)
{
	if(isImagePath(path))
	{
		paths.push_back(path);
	}
	else if(std::filesystem::is_directory(path))
	{
		for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator(path))
		{
			if(std::filesystem::is_directory(dirent.path()))
				getImageFiles(dirent.path(), paths);
			else if(isImagePath(dirent.path()))
				paths.push_back(dirent.path());
		}
	}
}

cv::Rect rectFromPoints(const std::vector<std::pair<cv::Point, int>>& points)
{
	int left = std::numeric_limits<int>::max();
	int right = std::numeric_limits<int>::min();
	int top = std::numeric_limits<int>::max();
	int bottom = std::numeric_limits<int>::min();

	for(const std::pair<cv::Point, int>& point : points)
	{
		left = point.first.x < left ? point.first.x : left;
		right = point.first.x > right ? point.first.x : right;

		top = point.first.y < top ? point.first.y : top;
		bottom = point.first.y > bottom ? point.first.y : bottom;
	}

	return cv::Rect(left, top, right-left, bottom-top);
}

double pointDist(const cv::Point2i& pointA, const cv::Point2i& pointB)
{
	cv::Vec2i a(pointA.x, pointA.y);
	cv::Vec2i b(pointB.x, pointB.y);
	return cv::norm(a-b);
}

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect)
{
	return point.x >= rect.x && point.x <= rect.x+rect.width &&
		   point.y >= rect.y && point.y <= rect.y+rect.height;
}
