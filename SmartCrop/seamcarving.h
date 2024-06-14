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

#include <opencv2/core/core.hpp>
#include <vector>

class SeamCarving
{
private:
	static cv::Mat GetEnergyImg(const cv::Mat &img);
	static cv::Mat computeGradientMagnitude(const cv::Mat &frame);
	static float intensity(float currIndex, int start, int end);
	static cv::Mat computePathIntensityMat(const cv::Mat &rawEnergyMap);
	static std::vector<int> getLeastImportantPath(const cv::Mat &importanceMap);
	static cv::Mat removeLeastImportantPath(const cv::Mat &original, const std::vector<int> &seam);
	static void removePixel(const cv::Mat &original, cv::Mat &outputMap, int row, int minCol);
	static cv::Mat addLeastImportantPath(const cv::Mat &original, const std::vector<int> &seam);
	static void addPixel(const cv::Mat &original, cv::Mat &outputMat, int row, int minCol);
	static cv::Mat drawSeam(const cv::Mat &frame, const std::vector<int> &seam);

public:
	static bool strechImage(cv::Mat& image, int seams, bool grow, std::vector<std::vector<int>>* seamsVect = nullptr);
	static bool strechImageVert(cv::Mat& image, int seams, bool grow, std::vector<std::vector<int>>* seamsVect = nullptr);
	static bool strechImageWithSeamsImage(cv::Mat& image, cv::Mat& seamsImage, int seams, bool grow);
};
