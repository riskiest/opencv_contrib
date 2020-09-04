// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_COLORSPACE_HPP__
#define __OPENCV_MCC_COLORSPACE_HPP__

#include <vector>
#include <string>
#include <iostream>
#include "io.hpp"
#include "operations.hpp"
//#include "utils.hpp"

namespace cv
{
namespace ccm
{

/* *\ brief Basic class for ColorSpace.
*/
class CV_EXPORTS ColorSpace
{
public:
    CV_PROP_RW IO io;
    CV_PROP_RW std::string type;
    CV_PROP_RW bool linear;
    CV_PROP_RW Operations to;
    CV_PROP_RW Operations from;
    CV_PROP_RW ColorSpace* l = 0;
    CV_PROP_RW ColorSpace* nl = 0;

    ColorSpace() {};

    ColorSpace(IO io_, std::string type_, bool linear_) :io(io_), type(type_), linear(linear_) {};

    virtual ~ColorSpace()
    {
        l = 0;
        nl = 0;
    };

    CV_WRAP virtual bool relate(const ColorSpace& other) const;

    CV_WRAP virtual Operations relation(const ColorSpace& /*other*/) const;

    bool operator<(const ColorSpace& other)const;
};

/* *\ brief Base of RGB color space;
   *        the argument values are from AdobeRGB;
   *        Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
*/
class CV_EXPORTS RGBBase_ : public ColorSpace
{
public:
    //primaries
    CV_PROP_RW double xr;
    CV_PROP_RW double yr;
    CV_PROP_RW double xg;
    CV_PROP_RW double yg;
    CV_PROP_RW double xb;
    CV_PROP_RW double yb;
    CV_PROP_RW MatFunc toL;
    CV_PROP_RW MatFunc fromL;
    CV_PROP_RW cv::Mat M_to;
    CV_PROP_RW cv::Mat M_from;

    using ColorSpace::ColorSpace;

    /* *\ brief There are 3 kinds of relationships for RGB:
       *        1. Different types;    - no operation
       *        1. Same type, same linear; - copy
       *        2. Same type, different linear, self is nonlinear; - 2 toL
       *        3. Same type, different linear, self is linear - 3 fromL
       *\ param other type of ColorSpace.
       *\ return Operations.
    */
    CV_WRAP Operations relation(const ColorSpace& other) const CV_OVERRIDE;

    /* *\ brief Initial operations.
    */
    CV_WRAP void init();

    /* *\ brief Produce color space instance with linear and non-linear versions.
       *\ param rgbl type of RGBBase_.
    */
    CV_WRAP void bind(RGBBase_& rgbl);

private:
    virtual void setParameter() {};

    /* *\ brief Calculation of M_RGBL2XYZ_base.
       *        see ColorSpace.pdf for details.
    */
    virtual void calM();

    /* *\ brief operations to or from XYZ.
    */
    virtual void calOperations();

    virtual void calLinear() {}

    virtual cv::Mat toLFunc(cv::Mat& /*rgb*/);

    virtual cv::Mat fromLFunc(cv::Mat& /*rgbl*/);

};

/* *\ brief Base of Adobe RGB color space;
*/
class CV_EXPORTS AdobeRGBBase_ : public RGBBase_
{
public:
    using RGBBase_::RGBBase_;
     double gamma;

private:
    cv::Mat toLFunc(cv::Mat& rgb) CV_OVERRIDE;
    cv::Mat fromLFunc(cv::Mat& rgbl) CV_OVERRIDE;
};

/* *\ brief Base of sRGB color space;
*/
class CV_EXPORTS sRGBBase_ : public RGBBase_
{
public:
    using RGBBase_::RGBBase_;
    CV_PROP_RW double a;
    CV_PROP_RW double gamma;
    CV_PROP_RW double alpha;
    CV_PROP_RW double beta;
    CV_PROP_RW double phi;
    CV_PROP_RW double K0;

private:
    /* *\ brief linearization parameters
       *        see ColorSpace.pdf for details;
    */
    virtual void calLinear() CV_OVERRIDE;

    /* *\ brief Used by toLFunc.
    */
    double toLFuncEW(double& x);

    /* *\ brief Linearization.
       *        see ColorSpace.pdf for details.
       *\ param rgb the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    cv::Mat toLFunc(cv::Mat& rgb) CV_OVERRIDE;

    /* *\ brief Used by fromLFunc.
    */
    double fromLFuncEW(double& x);

    /* *\ brief Delinearization.
       *        see ColorSpace.pdf for details.
       *\ param rgbl the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    cv::Mat fromLFunc(cv::Mat& rgbl) CV_OVERRIDE;
};

/* *\ brief sRGB color space.
   *        data from https://en.wikipedia.org/wiki/SRGB.
*/
class CV_EXPORTS sRGB_ :public sRGBBase_
{
public:
    sRGB_(bool linear_) :sRGBBase_(D65_2, "sRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief Adobe RGB color space.
*/
class CV_EXPORTS AdobeRGB_ : public AdobeRGBBase_
{
public:
    AdobeRGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "AdobeRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief Wide-gamut RGB color space.
   *        data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space.
*/
class CV_EXPORTS WideGamutRGB_ : public AdobeRGBBase_
{
public:
    WideGamutRGB_(bool linear_ = false) :AdobeRGBBase_(D50_2, "WideGamutRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief ProPhoto RGB color space.
   *        data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space.
*/
class CV_EXPORTS ProPhotoRGB_ : public AdobeRGBBase_
{
public:
    ProPhotoRGB_(bool linear_ = false) :AdobeRGBBase_(D50_2, "ProPhotoRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief DCI-P3 RGB color space.
   *        data from https://en.wikipedia.org/wiki/DCI-P3.
*/
class CV_EXPORTS DCI_P3_RGB_ : public AdobeRGBBase_
{
public:
    DCI_P3_RGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "DCI_P3_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief Apple RGB color space.
   *        data from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html.
*/
class CV_EXPORTS AppleRGB_ : public AdobeRGBBase_
{
public:
    AppleRGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "AppleRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief REC_709 RGB color space.
   *        data from https://en.wikipedia.org/wiki/Rec._709.
*/
class CV_EXPORTS REC_709_RGB_ : public sRGBBase_
{
public:
    REC_709_RGB_(bool linear_) :sRGBBase_(D65_2, "REC_709_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief REC_2020 RGB color space.
   *        data from https://en.wikipedia.org/wiki/Rec._2020.
*/
class CV_EXPORTS REC_2020_RGB_ : public sRGBBase_
{
public:
    REC_2020_RGB_(bool linear_) :sRGBBase_(D65_2, "REC_2020_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};


extern sRGB_ sRGB, sRGBL;
extern AdobeRGB_ AdobeRGB, AdobeRGBL;
extern WideGamutRGB_ WideGamutRGB, WideGamutRGBL;
extern ProPhotoRGB_ ProPhotoRGB, ProPhotoRGBL;
extern DCI_P3_RGB_ DCI_P3_RGB, DCI_P3_RGBL;
extern AppleRGB_ AppleRGB, AppleRGBL;
extern REC_709_RGB_ REC_709_RGB, REC_709_RGBL;
extern REC_2020_RGB_ REC_2020_RGB, REC_2020_RGBL;


/* *\ brief Bind RGB with RGBL.
*/
class CV_EXPORTS ColorSpaceInitial
{
public:
    CV_WRAP ColorSpaceInitial();
};

extern ColorSpaceInitial color_space_initial;

/* *\ brief Enum of the possible types of CAMs.
*/
enum CAM
{
    IDENTITY,
    VON_KRIES,
    BRADFORD
};

static std::map <std::tuple<IO, IO, CAM>, cv::Mat > cams;
const static cv::Mat Von_Kries = (cv::Mat_<double>(3, 3) << 0.40024, 0.7076, -0.08081, -0.2263, 1.16532, 0.0457, 0., 0., 0.91822);
const static cv::Mat Bradford = (cv::Mat_<double>(3, 3) << 0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296);
const static std::map <CAM, std::vector< cv::Mat >> MAs = {
    {IDENTITY , { cv::Mat::eye(cv::Size(3,3),CV_64FC1) , cv::Mat::eye(cv::Size(3,3),CV_64FC1)} },
    {VON_KRIES, { Von_Kries ,Von_Kries.inv() }},
    {BRADFORD, { Bradford ,Bradford.inv() }}
};

/* *\ brief XYZ color space.
   *        Chromatic adaption matrices.
*/
class CV_EXPORTS XYZ :public ColorSpace
{
public:
    XYZ(IO io_) : ColorSpace(io_, "XYZ", true) {};
    CV_WRAP Operations cam(IO dio, CAM method = BRADFORD);

private:
    /* *\ brief Get cam.
       *\ param sio the input IO of src.
       *\ param dio the input IO of dst.
       *\ param method type of CAM.
       *\ return the output array, type of cv::Mat.
    */
    cv::Mat cam_(IO sio, IO dio, CAM method = BRADFORD) const;
};

/* *\ brief Define XYZ_D65_2 and XYZ_D50_2.
*/
const XYZ XYZ_D65_2(D65_2);
const XYZ XYZ_D50_2(D50_2);


/* *\ brief Lab color space.
*/
class CV_EXPORTS Lab :public ColorSpace
{
public:
    Lab(IO io_);

private:
    static constexpr double delta = (6. / 29.);
    static constexpr double m = 1. / (3. * delta * delta);
    static constexpr double t0 = delta * delta * delta;
    static constexpr double c = 4. / 29.;

    cv::Vec3d fromxyz(cv::Vec3d& xyz);

    /* *\ brief Calculate From.
       *\ param src the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    cv::Mat fromsrc(cv::Mat& src);

    cv::Vec3d tolab(cv::Vec3d& lab);

    /* *\ brief Calculate To.
       *\ param src the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    cv::Mat tosrc(cv::Mat& src);
};

/* *\ brief Define Lab_D65_2 and Lab_D50_2.
*/
const Lab Lab_D65_2(D65_2);
const Lab Lab_D50_2(D50_2);

} // namespace ccm
} // namespace cv


#endif