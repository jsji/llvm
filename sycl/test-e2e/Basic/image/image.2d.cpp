// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- image.cpp - SYCL image basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

#include "../../helpers.hpp"

int main() {
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  constexpr size_t width = 4;
  const sycl::range<2> Img1Size(4, 4);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<float> out(width, 99.f);
  sycl::buffer<float, 1> buf((float *)out.data(), width);

  printf("Img1HostData: {");
  for (size_t i = 0; i < width; ++i)
    printf("{%f, %f, %f, %f}", Img1HostData[i].x(), Img1HostData[i].y(), Img1HostData[i].z(), Img1HostData[i].w());
  printf("}\n");

  {
    sycl::image<2> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    TestQueue Q{sycl::default_selector_v};
    const auto &d = Q.get_device();
    const std::string &name = d.get_info<sycl::info::device::name>();
    const std::string &driver_version = d.get_info<sycl::info::device::driver_version>();
    std::cout << "Device: " << name << "[" << driver_version << "]" << std::endl;
    float *array = (float *)malloc_shared(width * sizeof(float), Q);
    for (size_t i = 0; i < width; ++i)
      array[i] = 10.f;

    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto outAcc = buf.get_access<sycl::access_mode::write>(CGH, width);

      CGH.parallel_for<class ImgCopy>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        array[Item[0]] = Data[0];
        //outAcc[Item[0]] = Data[0];
        sycl::ext::oneapi::experimental::printf("\t\tItem[0] %zu, Item[1] %zu, Img1Acc %p, array %p\n", (size_t)Item[0], (size_t)Item[1], Img1Acc, array);
      });
    });
    Q.wait();

    printf("out: {");
    for (size_t i = 0; i < width; ++i)
      printf("%f, ", array[i]);
    printf("}\n");
  }

  std::cout << "Success" << std::endl;
  return 0;
}
