// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "precomp.hpp"

namespace cv
{
namespace ccm
{

double gammaCorrection_(const double& element, const double& gamma)
{
    return (element >= 0 ? pow(element, gamma) : -pow((-element), gamma));
}

cv::Mat gammaCorrection(const cv::Mat& src, const double& gamma)
{
    return elementWise(src, [gamma](double element)->double {return gammaCorrection_(element, gamma); });
}

cv::Mat maskCopyTo(const cv::Mat& src, const cv::Mat& mask)
{
    cv::Mat dst(countNonZero(mask), 1, src.type());
    const int channel = src.channels();
    auto it_mask = mask.begin<uchar>();
    switch (channel)
    {
    case 1:
    {
        auto it_src = src.begin<double>(), end_src = src.end<double>();
        auto it_dst = dst.begin<double>();
        for (; it_src != end_src; ++it_src, ++it_mask)
        {
            if (*it_mask)
            {
                (*it_dst) = (*it_src);
                ++it_dst;
            }
        }
        break;
    }
    case 3:
    {
        auto it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>();
        auto it_dst = dst.begin<cv::Vec3d>();
        for (; it_src != end_src; ++it_src, ++it_mask)
        {
            if (*it_mask)
            {
                (*it_dst) = (*it_src);
                ++it_dst;
            }
        }
        break;
    }
    default:
        throw std::invalid_argument { "Wrong channel!" };
        break;
    }
    return dst;
}

cv::Mat multiple(const cv::Mat& xyz, const cv::Mat& ccm)
{
    cv::Mat tmp = xyz.reshape(1, xyz.rows * xyz.cols);
    cv::Mat res = tmp * ccm;
    res = res.reshape(res.cols, xyz.rows);
    return res;
}

cv::Mat saturate(cv::Mat& src, const double& low, const double& up)
{
    cv::Mat dst = cv::Mat::ones(src.size(), CV_8UC1);
    cv::MatIterator_<cv::Vec3d> it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>();
    cv::MatIterator_<uchar> it_dst = dst.begin<uchar>();
    for (; it_src != end_src; ++it_src, ++it_dst)
    {
        for (int i = 0; i < 3; ++i)
        {
            if ((*it_src)[i] > up || (*it_src)[i] < low)
            {
                *it_dst = 0.;
                break;
            }
        }
    }
    return dst;
}

cv::Mat rgb2gray(cv::Mat rgb)
{
    return multiple(rgb, m_gray);
}

} // namespace ccm
} // namespace cv