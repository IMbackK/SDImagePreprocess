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
#include <exception>
#include <opencv2/core/mat.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include <filesystem>

class FaceRecognizer
{
public:

	struct Detection
	{
		int person;
		float confidence;
		cv::Rect rect;
	};

	class LoadException : public std::exception
	{
	private:
		std::string message;
	public:
		LoadException(const std::string& msg): std::exception(), message(msg) {}
		virtual const char* what() const throw() override
		{
			return message.c_str();
		}
	};

private:
	std::vector<cv::Mat> referanceFeatures;
	std::shared_ptr<cv::FaceRecognizerSF> recognizer;
	std::shared_ptr<cv::FaceDetectorYN> detector;

	double threshold = 0.363;

public:
	FaceRecognizer(std::filesystem::path recognizerPath = "", const std::filesystem::path& detectorPath = "", const std::vector<cv::Mat>& referances = std::vector<cv::Mat>());
	cv::Mat detectFaces(const cv::Mat& input);
	Detection isMatch(const cv::Mat& input, bool alone = false);
	bool addReferances(const std::vector<cv::Mat>& referances);
	void setThreshold(double threashold);
	double getThreshold();
	void clearReferances();
};
