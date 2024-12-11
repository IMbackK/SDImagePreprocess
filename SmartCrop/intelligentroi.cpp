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

#include "intelligentroi.h"

#include <opencv2/imgproc.hpp>

#include "utils.h"
#include "log.h"

bool InteligentRoi::compPointPrio(const std::pair<cv::Point2i, int>& a, const std::pair<cv::Point2i, int>& b, const cv::Point2i& center)
{
	if(a.second != b.second)
		return a.second > b.second;

	double distA = pointDist(a.first, center);
	double distB = pointDist(b.first, center);

	return distA < distB;
}

void InteligentRoi::slideRectToPoint(cv::Rect& rect, const cv::Point2i& point)
{
	if(!pointInRect(point, rect))
	{
		if(point.x < rect.x)
			rect.x = point.x;
		else if(point.x > rect.x+rect.width)
			rect.x = point.x-rect.width;
		if(point.y < rect.y)
			rect.y = point.y;
		else if(point.y > rect.y+rect.height)
			rect.y = point.y-rect.height;
	}
}

cv::Rect InteligentRoi::maxRect(bool& incompleate, const cv::Size2i& imageSize, double targetAspectRatio, std::vector<std::pair<cv::Point2i, int>> mustInclude)
{
	incompleate = false;

	cv::Point2i point(imageSize.width/2, imageSize.height/2);
	cv::Rect candiate;
	if(imageSize.width/targetAspectRatio > imageSize.height)
		candiate = cv::Rect(point.x-(imageSize.height*(targetAspectRatio/2)), 0, imageSize.height*targetAspectRatio, imageSize.height);
	else
		candiate = cv::Rect(0, point.y-(imageSize.width/targetAspectRatio)/2, imageSize.width, imageSize.width/targetAspectRatio);

	std::sort(mustInclude.begin(), mustInclude.end(),
		[&point](const std::pair<cv::Point2i, int>& a, const std::pair<cv::Point2i, int>& b){return compPointPrio(a, b, point);});

	while(true)
	{
		cv::Rect includeRect = rectFromPoints(mustInclude);
		if(includeRect.width-2 > candiate.width || includeRect.height-2 > candiate.height)
		{
			incompleate = true;
			slideRectToPoint(candiate, mustInclude.back().first);
			mustInclude.pop_back();
			Log(Log::DEBUG)<<"cant fill";
			for(const std::pair<cv::Point2i, int>& mipoint : mustInclude)
				Log(Log::DEBUG)<<mipoint.first<<' '<<pointDist(mipoint.first, point)<<' '<<mipoint.second;
		}
		else
		{
			break;
		}
	}

	for(const std::pair<cv::Point2i, int>& includePoint : mustInclude)
		slideRectToPoint(candiate, includePoint.first);

	if(candiate.x < 0)
		candiate.x = 0;
	if(candiate.y < 0)
		candiate.y = 0;
	if(candiate.x+candiate.width > imageSize.width)
		candiate.width = imageSize.width-candiate.x;
	if(candiate.y+candiate.height > imageSize.height)
		candiate.height = imageSize.height-candiate.y;

	return candiate;
}

InteligentRoi::InteligentRoi(const Yolo& yolo)
{
	personId = yolo.getClassForStr("person");
}

bool InteligentRoi::getCropRectangle(cv::Rect& out, const std::vector<Yolo::Detection>& detections, const cv::Size2i& imageSize, double targetAspectRatio)
{
	std::vector<std::pair<cv::Point2i, int>> corners;
	for(size_t i = 0; i < detections.size(); ++i)
	{
		int priority = detections[i].priority;
		if(priority > 0)
		{
			if(detections[i].class_id == personId)
			{
				corners.push_back({detections[i].box.tl()+cv::Point2i(detections[i].box.width/2, 0), priority+2});
				corners.push_back({detections[i].box.tl(), priority+1});
				corners.push_back({detections[i].box.br(), priority});
				corners.push_back({detections[i].box.tl()+cv::Point2i(detections[i].box.width, 0), priority+1});
				corners.push_back({detections[i].box.br()+cv::Point2i(0-detections[i].box.width, 0), priority});
			}
			else
			{
				corners.push_back({detections[i].box.tl(), priority});
				corners.push_back({detections[i].box.br(), priority});
				corners.push_back({detections[i].box.tl()+cv::Point2i(detections[i].box.width, 0), priority});
				corners.push_back({detections[i].box.br()+cv::Point2i(0-detections[i].box.width, 0), priority});
			}
		}
	}

	bool incompleate;
	out = maxRect(incompleate, imageSize, targetAspectRatio, corners);
	return incompleate;
}
