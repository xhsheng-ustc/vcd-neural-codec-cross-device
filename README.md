# Neural Codec — cross-device browser demo

Open the website: [Neural Codec cross-device demo](https://xhsheng-ustc.github.io/vcd-neural-codec-cross-device/)

Camera demo with CBR, resolution controls, bitrate and PSNR statistics, and comparisons with H.264, H.265, AV1, VP8, and MLVC.

The Neural Codec and MLVC each have separate GPU (WebGPU) and CPU (WebAssembly) encoder and decoder selectors. The Neural Codec defaults to a fast experimental cross-device runtime and also offers the fixed-point W16 runtime and the original floating-point runtime.

Open `index.html` through HTTPS or a local HTTP server. GitHub Pages can serve this directory directly from the repository root.
