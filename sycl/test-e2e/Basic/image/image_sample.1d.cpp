// REQUIRES: aspect-ext_intel_legacy_image
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Temporarily disable test on Windows due to regressions in GPU driver.
// UNSUPPORTED: hip, windows

#include <sycl/sycl.hpp>

#include <iostream>

class test_1d_class;
class test_2d_class;
class test_3d_class;

namespace s = sycl;

template <typename dataT>
bool check_result(dataT resultData, dataT expectedData, float epsilon = 0.1) {
  bool correct = true;
  if (std::abs(resultData.r() - expectedData.r()) > epsilon)
    correct = false;
  if (std::abs(resultData.g() - expectedData.g()) > epsilon)
    correct = false;
  if (std::abs(resultData.b() - expectedData.b()) > epsilon)
    correct = false;
  if (std::abs(resultData.a() - expectedData.a()) > epsilon)
    correct = false;
#ifdef DEBUG_OUTPUT
  if (!correct) {
    std::cout << "Expected: " << expectedData.r() << ", " << expectedData.g()
              << ", " << expectedData.b() << ", " << expectedData.a() << "\n";
    std::cout << "Got:      " << resultData.r() << ", " << resultData.g()
              << ", " << resultData.b() << ", " << resultData.a() << "\n";
  }
#endif // DEBUG_OUTPUT
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test1d_coord(dataT *hostPtr, coordT coord, dataT expectedColour) {
  dataT resultData;

  s::sampler testSampler(s::coordinate_normalization_mode::unnormalized,
                         s::addressing_mode::clamp, s::filtering_mode::linear);

  { // Scope everything to force destruction
    s::image<1> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<1>{3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    s::queue myQueue(s::default_selector_v);
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_1d_class>([=]() {
        dataT RetColor = imageAcc.read(coord, testSampler);
        resultDataAcc[0] = RetColor;
      });
    });
  }
  bool correct = check_result(resultData, expectedColour);
#ifdef DEBUG_OUTPUT
  if (!correct) {
    std::cout << "Coord: " << coord << "\n";
  }
#endif // DEBUG_OUTPUT
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test1d(coordT coord, dataT expectedResult) {
  dataT hostPtr[3];
  for (int i = 0; i < 3; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);
  return test1d_coord<dataT, coordT, channelType>(hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test2d(coordT coord, dataT expectedResult) {
  dataT hostPtr[9];
  for (int i = 0; i < 9; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);
  return test2d_coord<dataT, coordT, channelType>(hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test3d(coordT coord, dataT expectedResult) {
  dataT hostPtr[27];
  for (int i = 0; i < 27; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);
  return test3d_coord<dataT, coordT, channelType>(hostPtr, coord,
                                                  expectedResult);
}

int main() {

  bool passed = true;

  // 1d image read tests
  if (!test1d<s::float4, float, s::image_channel_type::fp32>(
          0.0f, s::float4(0, 10, 20, 30)))
    passed = false;
  return passed ? 0 : -1;
}
