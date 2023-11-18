# ESRGAN-candle-rs

ESRGAN implemented in rust with candle.

This was an experiment I did to try out the candle library and compare it with pytorch using a model architecture I'm very familiar with. I implemented both the standard ESRGAN arch as well as the RealESRGAN version with the different key names.

It ended up being slower in general than pytorch, and I created an issue in the candle repository as a result. This repo is just to share that code.

This code isn't meant to be directly runnable as some kind of CLI app. It requires you to modify some paths in the main.rs file. You'll also have to manually change the params of the model based on whichever you want to run.

Models must be converted to .safetensors format before they can be used. This can be done with a simple python script or via chaiNNer.