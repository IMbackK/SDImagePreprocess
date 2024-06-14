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

#include <filesystem>
#include <vector>
#include <opencv2/imgproc.hpp>

bool isImagePath(const std::filesystem::path& path);

void getImageFiles(const std::filesystem::path& path, std::vector<std::filesystem::path>& paths);

cv::Rect rectFromPoints(const std::vector<std::pair<cv::Point, int>>& points);

double pointDist(const cv::Point2i& pointA, const cv::Point2i& pointB);

bool pointInRect(const cv::Point2i& point, const cv::Rect& rect);
