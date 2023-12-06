# ESRGAN-candle-rs

ESRGAN implemented in rust with candle.

This was an experiment I did to try out the candle library and compare it with pytorch using a model architecture I'm very familiar with. I implemented both the standard ESRGAN arch as well as both the RealESRGAN version with the different key names and the RealESRGANv2 SRVGGCompact arch (aka Compact).

It ended up being slower in general than pytorch, and I created an issue in the candle repository as a result. This repo is just to share that code.

This code wasn't originally meant to be directly runnable as a CLI app. I'm currently working towards making it into one, but for now it is a bit limited. Models must be converted to .safetensors format before they can be used. This can be done with a simple python script (one is provided in this repo) or via chaiNNer.

## CLI usage

Here is a basic usage example:

```cargo run -- --model ".\path\to\the\model.safetensors" -i ".\input" -o ".\output" --device 0```

## Models

The official RealESRGAN x4 model can be found [here](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth).

Community trained models can be found [here](https://openmodeldb.info/?t=arch%3Aesrgan).

This project contains automatic parameter detection for scale, in_nc, out_nc, num_filters, and num_blocks -- but only for old-arch esrgan models. If you wish to use a new-arch esrgan model with parameters other than default, either load and save the model using chaiNNer (they get auto-converted to old-arch there), or use the CLI args to manually change the arch type and parameters.
