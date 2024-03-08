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

  constexpr size_t width = 4;
  const sycl::range<1> Img1Size(4);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<float> out(width);
  sycl::buffer<float, 1> buf((float *)out.data(), width);
  {
    sycl::image<1> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    TestQueue Q{sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto outAcc = buf.get_access<sycl::access_mode::write>(CGH, width);

      CGH.parallel_for<class ImgCopy>(width, [=](sycl::id<1> id) {
        sycl::float4 Data = Img1Acc.read((size_t)id[0]);
        outAcc[id[0]] = Data[0];
      });
    });
  }

  std::cout << "Success" << std::endl;
  return 0;
}
