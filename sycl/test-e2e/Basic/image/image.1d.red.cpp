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
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::r;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::unorm_int8;

  constexpr auto SYCLRead = sycl::access::mode::read;

  constexpr size_t width = 4;
  const sycl::range<1> Img1Size(4);

  std::vector<int8_t> Img1HostData(Img1Size.size(), {1});
  std::vector<int8_t> out(width);
  sycl::buffer<int8_t, 1> buf((int8_t *)out.data(), width);
  {
    sycl::image<1> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    TestQueue Q{sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<int8_t, SYCLRead>(CGH);
      auto outAcc = buf.get_access<sycl::access_mode::write>(CGH, width);

      CGH.parallel_for<class ImgCopy>(width, [=](sycl::id<1> id) {
        int8_t Data = Img1Acc.read((size_t)id[0]);
        outAcc[id[0]] = Data;
      });
    });
  }

  std::cout << "Success" << std::endl;
  return 0;
}
