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

#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class Yolo
{
public:
	struct Detection
	{
		int class_id = 0;
		std::string className;
		float confidence = 0.0;
		int priority = -1;
		cv::Scalar color;
		cv::Rect box;
	};

private:
	static constexpr float modelConfidenceThreshold = 0.20;
	static constexpr float modelScoreThreshold = 0.40;
	static constexpr float modelNMSThreshold = 0.45;

	std::string modelPath;
	std::vector<std::pair<std::string, int>> classes;
	cv::Size2f modelShape;
	bool letterBoxForSquare = true;
	cv::dnn::Net net;

	void loadClasses(const std::string& classes);
	void loadOnnxNetwork(const std::filesystem::path& path);
	cv::Mat formatToSquare(const cv::Mat &source);
	static void clampBox(cv::Rect& box, const cv::Size& size);

public:
	Yolo(const std::filesystem::path &onnxModelPath = "", const cv::Size& modelInputShape = {640, 480},
		const std::filesystem::path& classesTxtFilePath = "", bool runWithOCl = true);
	std::vector<Detection> runInference(const cv::Mat &input);
	int getClassForStr(const std::string& str) const;
};
