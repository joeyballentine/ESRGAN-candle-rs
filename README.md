# ESRGAN-candle-rs

ESRGAN implemented in rust with candle.

This was an experiment I did to try out the candle library and compare it with pytorch using a model architecture I'm very familiar with. I implemented both the standard ESRGAN arch as well as the RealESRGAN version with the different key names.

It ended up being slower in general than pytorch, and I created an issue in the candle repository as a result. This repo is just to share that code.

This code wasn't originally meant to be directly runnable as a CLI app. I'm currently working towards making it into one, but for now it is a bit limited. Right now, to run anything but standard config RealESRGAN-style models, you'll have to manually change the params of the model based on whichever you want to run.

Models must be converted to .safetensors format before they can be used. This can be done with a simple python script or via chaiNNer.

## CLI usage

Here is a basic usage example:

```cargo run -- --model ".\path\to\the\model.safetensors" -i ".\input" -o ".\output" --device 0```