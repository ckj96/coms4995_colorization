# End-to-End Fully Automatic Colorization
The model is based on [Let there be Color!: Automatic Colorization of Grayscale Images](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/)


## Overview

This code provides an implementation of the research paper:

```
  "Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification"
  Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa
  ACM Transaction on Graphics (Proc. of SIGGRAPH 2016), 2016
```

We learn to automatically color grayscale images with a deep network. Our
network learns both local features and global features jointly in a single
framework. Our approach can then be used on images of any resolution. By
incorporating global features we are able to obtain realistic colorings with
our model.

## License

```
  Copyright (C) <2016> <Satoshi Iizuka, Edgar Simo-Serra, Hiroshi Ishikawa>

  This work is licensed under the Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy
  of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or
  send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

  Satoshi Iizuka, Waseda University
  iizuka@aoni.waseda.jp, http://hi.cs.waseda.ac.jp/~iizuka/index_eng.html
  Edgar Simo-Serra, Waseda University
  esimo@aoni.waseda.jp, http://hi.cs.waseda.ac.jp/~esimo/  
```


## Dependencies

- [Torch7](http://torch.ch/docs/getting-started.html)
- [nn](https://github.com/torch/nn)
- [image](https://github.com/torch/image)
- [nngraph](https://github.com/torch/nngraph)

All packages should be part of a standard Torch7 install. For information on how to install Torch7 please see the [official torch documentation](http://torch.ch/docs/getting-started.html) on the subject.

## Usage

First, download the colorization model by running the download script:

```
./download_model.sh
```

Basic usage is:

```
th colorize.lua <input_image> [<output_image>]
```

For example:

```
th colorize.lua ansel_colorado_1941.png out.png
```

### Best Performance

- This model was trained on the [Places dataset](http://places.csail.mit.edu/) and thus best performance is for natural outdoor images.
- While the model works on any size image, we trained it on 224x224 pixel images and thus it works best on small images. Note that you can process a small imageto obtain the chrominance map and then rescale it and combine it with the original grayscale image for higher quality.
- Larger image sizes can give uneven colorings (limited by spatial support of the network).

### ImageNet Model
We also provide the colorization model that was trained on [ImageNet](http://image-net.org/challenges/LSVRC/2012/index). This model can be used for comparisons with other colorization models trained on ImageNet. We recommend using the places colorization model for general purposes.

For using the ImageNet model, download the model by running:

```
./download_model_imagenet.sh
```

Usage is:

```
th colorize.lua <input_image> <output_image> colornet_imagenet.t7
```

## Acknowledgement
The model credits to Yang Liu, Alibaba-Zhejiang University Joint Research Institute of Frontier Technologies








