Host `Intel(R) Xeon(R) Processor`, 4 logical CPUs at 2400.0 MHz, MemTotal 16387328 kB. ONNX Runtime 1.32.0, intra-op threads 4, input ['batch', 3, 'height', 'width']. Model sha256 `c7e13e3fe57a28b5…`. 10 warmup iterations then 40 measured iterations per image. Percentiles use NumPy's linear method. Values below are shown to 4 decimal places; exact floats are in `benchmarks/benchmark.json`. Every iteration letterboxes the photo to 640×640 before the forward pass.

| Image | Pre p50 | Pre p95 | Infer p50 | Infer p95 | Post p50 | Post p95 |
|---|---:|---:|---:|---:|---:|---:|
| bus.jpg | 1.6942 | 1.7658 | 19.2595 | 20.1809 | 0.7563 | 0.8598 |
| zidane.jpg | 1.3753 | 1.4832 | 19.2655 | 19.5779 | 0.6837 | 0.7195 |
| all images | 1.5304 | 1.7517 | 19.2647 | 20.1619 | 0.7349 | 0.7941 |

Sample clip `app/static/street.mp4`: 120 frames, source 15.0 fps. Pipeline FPS (frames / sum of preprocess+inference+postprocess) **42.2275**. Wall-clock FPS of that same loop **41.9929**.

| Run | Infer p50 | Infer p95 | Clip pipeline FPS |
|---|---:|---:|---:|
| 4 threads, 640 | 19.2647 | 20.1619 | 42.2275 |
| 1 thread, 640 | 50.5584 | 50.8580 | — |
