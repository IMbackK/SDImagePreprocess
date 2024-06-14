/**
* Lubricant Detecter
* Copyright (C) 2021 Carl Klemm
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* version 3 as published by the Free Software Foundation.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the
* Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
* Boston, MA  02110-1301, USA.
*/

#include "log.h"

Log::Log(Level type, bool endlineI): endline(endlineI)
{
	msglevel = type;
	if(headers)
	{
		operator << ("["+getLabel(type)+"] ");
	}
}

Log::~Log()
{
	if(opened && endline)
	{
		std::cout<<'\n';
	}
	opened = false;
}


std::string Log::getLabel(Level level)
{
	std::string label;
	switch(level)
	{
		case DEBUG:
			label = "DEBUG";
			break;
		case INFO:
			label = "INFO ";
			break;
		case WARN:
			label = "WARN ";
			break;
		case ERROR:
			label = "ERROR";
			break;
	}
	return label;
}

bool Log::headers = false;
Log::Level Log::level = WARN;
