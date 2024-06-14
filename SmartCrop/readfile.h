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
#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sstream>

inline std::string readFile(const std::filesystem::path& path)
{
	std::ifstream file(path);
	if(!file.is_open())
		throw std::runtime_error(std::string("could not open file ") + path.string());
	std::stringstream ss;
	ss<<file.rdbuf();
	return ss.str();
}
