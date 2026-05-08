A lightweight Convolutional Neural Network (CNN) implemented in PyTorch for upscaling game images from 720p to 1440p (2x Super Resolution). 

The repository provides two variants:
- **Fast Version:** Features 4 residual blocks and a lower channel count for real-time applications.
- **Quality Version:** Features 16 residual blocks for higher fidelity upscaling.

Additionally, this repository introduces **LoRA (Low-Rank Adaptation)** support for the CNN blocks. You can efficiently fine-tune the super-resolution network for specific video games or styles with a fraction of the trainable parameters.
