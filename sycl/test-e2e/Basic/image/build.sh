clang++ -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=off -fno-sycl-instrument-device-code -Xclang -fsycl-disable-range-rounding -Xs "-device dg2-g10-c0" -w        image.3d.cpp
