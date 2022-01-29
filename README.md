# Large Scale Object Detection using SAHI and DETIC (Detectron2)

## Sources

- DETIC Paper: https://arxiv.org/abs/2201.02605
- DETIC GitHub: https://github.com/facebookresearch/Detic
- DETIC Explainer Video: https://www.youtube.com/watch?v=Xw2icCN5ZpM
- DETIC Quick Start in Studiolab: https://github.com/machinelearnear/detic-detecting-20k-classes-using-image-level-supervision
- SAHI: https://github.com/obss/sahi

## What is DETIC?

Detic: A Detector with image classes that can use image-level labels to easily train detectors. This means you can use more than 20k classes from ImageNet 21K or combine it with CLIP embeddings to expand to any number of classes. Please have a look at the [original repo](https://github.com/facebookresearch/Detic) for more information and check my [YouTube Video](https://www.youtube.com/watch?v=Xw2icCN5ZpM) and [Repo](https://github.com/machinelearnear/detic-detecting-20k-classes-using-image-level-supervision) to get started easily.

### Features
- Detects any class given class names (using CLIP).
- We train the detector on ImageNet-21K dataset with 21K classes.
- Cross-dataset generalization to OpenImages and Objects365 without finetuning.
- State-of-the-art results on Open-vocabulary LVIS and Open-vocabulary COCO.
- Works for DETR-style detectors.

## What is SAHI?

> Object detection and instance segmentation are by far the most important fields of applications in Computer Vision. 
> However, detection of small objects and inference on large images are still major issues in practical usage. 
> Here comes the SAHI to help developers overcome these real-world problems with many vision utilities.

> <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img width="600" src="https://user-images.githubusercontent.com/34196005/144092739-c1d9bade-a128-4346-947f-424ce00e5c4f.gif" alt="sahi-yolox"></a> 

Please go to the [original repo](https://github.com/obss/sahi) for more information. 

## Why this repo?

This is a quick example that should get you started with the basics of SAHI combined with the flexibility that DETIC gives you to choose whatever set of classes you need for your use case without the need for any fine-tuning. This might be very useful to test an idea and get a project off the ground.

This is the snippet of code that defines the `Detectron2DeticModel`

```python
detection_model = Detectron2DeticModel(
    model_path='https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth',
    config_path='configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml',
    confidence_threshold=0.5,
    vocabulary='custom',
    custom_classes=['car','truck','automobile','sign','traffic light'],
    device="cuda",
)
```

And this, what you need to run to get a sliced prediction

```python
result = get_sliced_prediction(
    im,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)
```

## Note

- Everything that you see here is not optimised for PROD workloads and should be taken with a pinch of salt. Use under your own discretion.

## References

```bibtext
@inproceedings{zhou2021detecting,
  title={Detecting Twenty-thousand Classes using Image-level Supervision},
  author={Zhou, Xingyi and Girdhar, Rohit and Joulin, Armand and Kr{\"a}henb{\"u}hl, Philipp and Misra, Ishan},
  booktitle={arXiv preprint arXiv:2201.02605},
  year={2021}
}
```

```bibtext
@software{akyon2021sahi,
  author       = {Akyon, Fatih Cagatay and Cengiz, Cemil and Altinuc, Sinan Onur and Cavusoglu, Devrim and Sahin, Kadir and Eryuksel, Ogulcan},
  title        = {{SAHI: A lightweight vision library for performing large scale object detection and instance segmentation}},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5718950},
  url          = {https://doi.org/10.5281/zenodo.5718950}
}
```