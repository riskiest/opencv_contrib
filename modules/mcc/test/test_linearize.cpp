// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

Mat s = (Mat_<Vec3d>(24, 1) <<
        Vec3d(214.11, 98.67, 37.97),
        Vec3d(231.94, 153.1, 85.27),
        Vec3d(204.08, 143.71, 78.46),
        Vec3d(190.58, 122.99, 30.84),
        Vec3d(230.93, 148.46, 100.84),
        Vec3d(228.64, 206.97, 97.5),
        Vec3d(229.09, 137.07, 55.29),
        Vec3d(189.21, 111.22, 92.66),
        Vec3d(223.5, 96.42, 75.45),
        Vec3d(201.82, 69.71, 50.9),
        Vec3d(240.52, 196.47, 59.3),
        Vec3d(235.73, 172.13, 54.),
        Vec3d(131.6, 75.04, 68.86),
        Vec3d(189.04, 170.43, 42.05),
        Vec3d(222.23, 74., 71.95),
        Vec3d(241.01, 199.1, 61.15),
        Vec3d(224.99, 101.4, 100.24),
        Vec3d(174.58, 152.63, 91.52),
        Vec3d(248.06, 227.69, 140.5),
        Vec3d(241.15, 201.38, 115.58),
        Vec3d(236.49, 175.87, 88.86),
        Vec3d(212.19, 133.49, 54.79),
        Vec3d(181.17, 102.94, 36.18),
        Vec3d(115.1, 53.77, 15.23));
s = s / 255;
double gamma = 2.2;
int deg = 3;
Color color = Macbeth_D50_2;
color.getGray();
RGBBase_ cs = sRGB;
Mat mask = saturate(s, 0.05, 0.93);

TEST(CV_ccmLinearize, test_identity)
{
    Mat y = (Mat_<Vec3d>(24, 1) <<
        Vec3d(0.83964706, 0.38694118, 0.14890196),
        Vec3d(0.90956863, 0.60039216, 0.33439216),
        Vec3d(0.80031373, 0.56356863, 0.30768627),
        Vec3d(0.74737255, 0.48231373, 0.12094118),
        Vec3d(0.90560784, 0.58219608, 0.39545098),
        Vec3d(0.89662745, 0.81164706, 0.38235294),
        Vec3d(0.89839216, 0.53752941, 0.21682353),
        Vec3d(0.742, 0.43615686, 0.36337255),
        Vec3d(0.87647059, 0.37811765, 0.29588235),
        Vec3d(0.79145098, 0.27337255, 0.19960784),
        Vec3d(0.94321569, 0.77047059, 0.23254902),
        Vec3d(0.92443137, 0.67501961, 0.21176471),
        Vec3d(0.51607843, 0.29427451, 0.27003922),
        Vec3d(0.74133333, 0.66835294, 0.16490196),
        Vec3d(0.8714902, 0.29019608, 0.28215686),
        Vec3d(0.94513725, 0.78078431, 0.23980392),
        Vec3d(0.88231373, 0.39764706, 0.39309804),
        Vec3d(0.68462745, 0.59854902, 0.35890196),
        Vec3d(0.97278431, 0.89290196, 0.55098039),
        Vec3d(0.94568627, 0.78972549, 0.4532549),
        Vec3d(0.92741176, 0.68968627, 0.34847059),
        Vec3d(0.83211765, 0.5234902 , 0.21486275),
        Vec3d(0.71047059, 0.40368627, 0.14188235),
        Vec3d(0.45137255, 0.21086275, 0.05972549));

    ASSERT_MAT_NEAR(LinearIdentity().linearize(s), y, 1e-4);
}

TEST(CV_ccmLinearize, test_gamma)
{
    Mat y = (Mat_<Vec3d>(24, 1) <<
        Vec3d(0.68078957, 0.12382801, 0.01514889),
        Vec3d(0.81177942, 0.32550452, 0.089818),
        Vec3d(0.61259378, 0.2831933 , 0.07478902),
        Vec3d(0.52696493, 0.20105976, 0.00958657),
        Vec3d(0.80402284, 0.30419523, 0.12989841),
        Vec3d(0.78658646, 0.63184111, 0.12062068),
        Vec3d(0.78999637, 0.25520249, 0.03462853),
        Vec3d(0.51866697, 0.16114393, 0.1078387),
        Vec3d(0.74820768, 0.11770076, 0.06862177),
        Vec3d(0.59776825, 0.05765816, 0.02886627),
        Vec3d(0.8793145, 0.56346033, 0.0403954),
        Vec3d(0.84124847, 0.42120746, 0.03287592),
        Vec3d(0.23333214, 0.06780408, 0.05612276),
        Vec3d(0.5176423, 0.41210976, 0.01896255),
        Vec3d(0.73888613, 0.06575388, 0.06181293),
        Vec3d(0.88326036, 0.58018751, 0.04321991),
        Vec3d(0.75922531, 0.13149072, 0.1282041),
        Vec3d(0.4345097, 0.32331019, 0.10494139),
        Vec3d(0.94110142, 0.77941419, 0.26946323),
        Vec3d(0.88438952, 0.5949049 , 0.17536928),
        Vec3d(0.84722687, 0.44160449, 0.09834799),
        Vec3d(0.66743106, 0.24076803, 0.03394333),
        Vec3d(0.47141286, 0.13592419, 0.01362205),
        Vec3d(0.17377101, 0.03256864, 0.00203026));

    ASSERT_MAT_NEAR(LinearGamma(gamma).linearize(s), y, 1e-4);
}

TEST(CV_ccmLinearize, test_color_polyfit)
{
    Mat y = (Mat_<Vec3d>(24, 1) <<
        Vec3d(2.63480590e-01, 8.19784230e-02, 5.88380570e-02),
        Vec3d(5.00308824e-01, 2.55670153e-01, 2.84774400e-01),
        Vec3d(1.65547676e-01, 2.18326729e-01, 2.34834241e-01),
        Vec3d(6.91579431e-02, 1.46243551e-01, 4.85506219e-02),
        Vec3d(4.84584944e-01, 2.36873580e-01, 4.21828895e-01),
        Vec3d(4.49999903e-01, 5.15593650e-01, 3.89716261e-01),
        Vec3d(4.56680121e-01, 1.93624454e-01, 1.09260805e-01),
        Vec3d(6.14667752e-02, 1.12219287e-01, 3.45822331e-01),
        Vec3d(3.77645006e-01, 7.72485552e-02, 2.14671313e-01),
        Vec3d(1.46690033e-01, 3.83676678e-02, 9.30261311e-02),
        Vec3d(6.45841448e-01, 4.59722537e-01, 1.26168624e-01),
        Vec3d(5.61924874e-01, 3.39304873e-01, 1.04244341e-01),
        Vec3d(-1.24921202e-03, 4.34819857e-02, 1.74574965e-01),
        Vec3d(6.05378564e-02, 3.31429676e-01, 6.74475908e-02),
        Vec3d(3.60866062e-01, 4.23768038e-02, 1.92686670e-01),
        Vec3d(6.54814616e-01, 4.73515714e-01, 1.34642338e-01),
        Vec3d(3.97880165e-01, 8.80047529e-02, 4.15950016e-01),
        Vec3d(8.88761384e-04, 2.53736288e-01, 3.35935826e-01),
        Vec3d(7.92140863e-01, 6.31444183e-01, 9.20127919e-01),
        Vec3d(6.57391804e-01, 4.85584457e-01, 5.81564694e-01),
        Vec3d(5.74784077e-01, 3.56891943e-01, 3.13534251e-01),
        Vec3d(2.42877023e-01, 1.80918764e-01, 1.07292088e-01),
        Vec3d(2.34750448e-02, 9.15416417e-02, 5.56885760e-02),
        Vec3d(4.16360011e-02, 3.14799517e-02, 4.67810688e-02));

    ASSERT_MAT_NEAR(LinearColor<Polyfit>(deg, s, color, mask, cs).linearize(s), y, 1e-4);
}

TEST(CV_ccmLinearize, test_color_logpolyfit)
{
    Mat y = (Mat_<Vec3d>(24, 1) <<
        Vec3d(0.21210199, 0.08042872, 0.06177358),
        Vec3d(0.43902276, 0.25803696, 0.22625212),
        Vec3d(0.1392843, 0.21910892, 0.16649895),
        Vec3d(0.07929871, 0.14429388, 0.06124959),
        Vec3d(0.42175787, 0.23847603, 0.50200816),
        Vec3d(0.38486859, 0.49908647, 0.41903521),
        Vec3d(0.39187715, 0.19330825, 0.07678465),
        Vec3d(0.07496683, 0.10997862, 0.3253915),
        Vec3d(0.31256906, 0.07589457, 0.14682645),
        Vec3d(0.12666458, 0.03865245, 0.07035133),
        Vec3d(0.61317011, 0.453896, 0.08471713),
        Vec3d(0.509618, 0.34295792, 0.07466823),
        Vec3d(0.01792609, 0.04375426, 0.11416985),
        Vec3d(0.07444757, 0.33517131, 0.06308572),
        Vec3d(0.29675398, 0.04267084, 0.1279182),
        Vec3d(0.62473933, 0.46546083, 0.08914635),
        Vec3d(0.33211973, 0.08623901, 0.48580852),
        Vec3d(0.04223237, 0.25602914, 0.30707565),
        Vec3d(0.81340438, 0.57786249, 5.63433098),
        Vec3d(0.62807965, 0.47536376, 1.17181577),
        Vec3d(0.52493661, 0.36014609, 0.26894465),
        Vec3d(0.19575899, 0.18007616, 0.0759408),
        Vec3d(0.05430091, 0.08966718, 0.06150038),
        Vec3d(0.02983422, 0.03045434, 0.03305337));

    ASSERT_MAT_NEAR(LinearColor<LogPolyfit>(deg, s, color, mask, cs).linearize(s), y, 1e-4);
}

TEST(CV_ccmLinearize, test_gray_polyfit)
{
    Mat y = (Mat_<Vec3d>(24, 1) <<
        Vec3d(0.51666899, 0.05119734, 0.08355961),
        Vec3d(0.59846831, 0.22746696, 0.03275719),
        Vec3d(0.46851612, 0.18821734, 0.02831637),
        Vec3d(0.40274707, 0.11294869, 0.10983447),
        Vec3d(0.59401581, 0.20773439, 0.05530918),
        Vec3d(0.58382767, 0.48249077, 0.04910519),
        Vec3d(0.58583962, 0.16222916, 0.04147151),
        Vec3d(0.39606823, 0.07893876, 0.04140637),
        Vec3d(0.56052673, 0.0472528, 0.02748943),
        Vec3d(0.45754742, 0.02793019, 0.04935323),
        Vec3d(0.63517156, 0.43148826, 0.0358448),
        Vec3d(0.61493813, 0.31386138, 0.04359845),
        Vec3d(0.14207934, 0.02743228, 0.02822652),
        Vec3d(0.39523985, 0.30585866, 0.07094004),
        Vec3d(0.55468508, 0.02734776, 0.0274375),
        Vec3d(0.6372021, 0.44431108, 0.03374266),
        Vec3d(0.56733935, 0.05641847, 0.05414246),
        Vec3d(0.32546855, 0.22544012, 0.0398225),
        Vec3d(0.66553671, 0.57956479, 0.17545548),
        Vec3d(0.63778086, 0.45540855, 0.09069633),
        Vec3d(0.61819338, 0.33161218, 0.03647687),
        Vec3d(0.50753594, 0.14890516, 0.0422774),
        Vec3d(0.35705032, 0.05956936, 0.08964481),
        Vec3d(0.0893518, 0.04399409, 0.18717526));

    ASSERT_MAT_NEAR(LinearGray<Polyfit>(deg, s, color, mask, cs).linearize(s), y, 1e-4);
}

TEST(CV_ccmLinearize, test_gray_logpolyfit)
{
    Mat y = (Mat_<Vec3d>(24, 1) <<
        Vec3d(4.60331981e-01, 5.51120816e-02, 3.97365482e-01),
        Vec3d(4.73679407e-01, 2.29393836e-01, 3.84195886e-02),
        Vec3d(4.38764462e-01, 1.88051166e-01, 3.33576233e-02),
        Vec3d(3.95560026e-01, 1.11621603e-01, 5.06142257e+00),
        Vec3d(4.73785486e-01, 2.08586471e-01, 5.87127685e-02),
        Vec3d(4.73648490e-01, 4.45971601e-01, 5.32792915e-02),
        Vec3d(4.73716999e-01, 1.61001724e-01, 4.02782927e-02),
        Vec3d(3.90399630e-01, 7.97334727e-02, 4.64809651e-02),
        Vec3d(4.71407192e-01, 5.16538838e-02, 3.18491932e-02),
        Vec3d(4.32594201e-01, 3.03426453e-02, 5.36226594e-02),
        Vec3d(4.68788432e-01, 4.16218138e-01, 3.42954435e-02),
        Vec3d(4.72387015e-01, 3.17700868e-01, 4.32262905e-02),
        Vec3d(1.40435644e-01, 3.16799217e-02, 3.02925240e-02),
        Vec3d(3.89750609e-01, 3.09885279e-01, 1.63338781e-01),
        Vec3d(4.70437333e-01, 3.12909236e-02, 3.07003735e-02),
        Vec3d(4.68300410e-01, 4.24570259e-01, 3.26637048e-02),
        Vec3d(4.72334300e-01, 5.96850848e-02, 5.76907392e-02),
        Vec3d(3.28844114e-01, 2.27258414e-01, 4.50583791e-02),
        Vec3d(4.58942833e-01, 4.73436696e-01, 1.74710299e-01),
        Vec3d(4.68156974e-01, 4.31339830e-01, 9.05259853e-02),
        Vec3d(4.71960375e-01, 3.34642678e-01, 4.19957551e-02),
        Vec3d(4.56964629e-01, 1.47352730e-01, 4.13462335e-02),
        Vec3d(3.57831381e-01, 6.24515609e-02, 6.54872609e-01),
        Vec3d(8.92793201e-02, 4.38221622e-02, 3.22837394e+08));

    ASSERT_MAT_NEAR(LinearGray<LogPolyfit>(deg, s, color, mask, cs).linearize(s)(cv::Rect(0, 0, 1, 23)), y(cv::Rect(0, 0, 1, 23)), 1e-4);
}

} // namespace
} // namespace opencv_test