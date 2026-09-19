# RK3576 random inference benchmark

This is a standalone C++ RKNN Runtime application for the quantized Sonic
encoder and decoder. It runs the full synthetic path:

```text
random encoder obs [1247] -> encoder token [64]
                              ↓ overwrite decoder token_state
random decoder obs [994] -> decoder action [29]
```

The benchmark reports average/p50/p99 latency and average Hz for each model
and the end-to-end pair. It measures NPU inference plus RKNN input/output
transfers, not random-number generation.

On the RK3576 board:

```bash
g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude src/main.cpp \
  -L/usr/lib -lrknnrt -Wl,-rpath,/usr/lib -o sonic_rk3576_infer
./sonic_rk3576_infer \
  --encoder /home/soulde/sonic_rk3576_int8_20260919/sonic_encoder_int8.rknn \
  --decoder /home/soulde/sonic_rk3576_int8_20260919/sonic_decoder_int8.rknn \
  --warmup 100 --iterations 1000
```

The random inputs are only for throughput/smoke testing. They are not a
control-policy accuracy test.

The measured 5,000-iteration result on the current RK3576 board was 268.75 Hz
for encoder, 232.89 Hz for decoder, and 124.75 Hz end-to-end, with 8.37 ms
end-to-end p99 latency.
