# AI in Logistics: Optimizing Container Fill Rate with Computer Vision

[[`Project Writeup`](https://medium.com/@jonathanlawhh) [`My Website`](https://jonathanlawhh.com/)]

## Project Overview
![Image of a container and the prediction](/assets/fill_rate_demo_01.png)
This project explores the use of Segment Anything Model (SAM) by Meta researchers and computer vision to detect and calculate container fill rate based on pallet segmentation. It is designed to demonstrate the potential of this approach for optimizing logistics operations.

This project takes into assumption that the fill rate will be calculated at each layer of the container loading process. The project can then be further enhanced to read from a live stream.


## References

- [Segment Anything Model](https://segment-anything.com/) by Meta researchers
- [OpenCV](https://opencv.org/)

## Setup and Usage

### Hardware Requirements
- GPU with CUDA support (recommended for optimal performance)
- CPU is supported but may result in slower processing times (30 seconds to 2 minutes per image)

### Software Requirements
- Python 3.10 (specific version required for PyTorch compatibility at the time of writing)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/jonathanlawhh/container-fill-rate-ai.git
```
2. Install required libraries:
```bash
pip install -R requirements.txt
```
3. Download the SAM checkpoint:
   - Download the vit_l - ViT-L SAM model checkpoint from Meta's SAM GitHub page: https://github.com/facebookresearch/segment-anything#model-checkpoints.
   - Place the downloaded checkpoint file in the .\models\ folder.

### Usage

1. Place your container images in the .\data\ folder.

2. Run the script.
```bash
python main.py
```

## Closing thoughts

- An SSD (Single Shot Detection) or YOLO (You Only Look Once) model could be used as an alternative approach to detect pallets.
- Pallets in the back layer may be captured as the front layer, resulting in false "filled" space.
- In this approach using pallet labels as prompting points, the pro is that the chances of trash not being segmented are higher.
- Production readiness: Depends on the environment it is to be deployed in and whether it can be customized accordingly.
  - Ideally, a fixed camera on top of the docking area, pointing into the container, would be ideal.
  - Handheld photographs require the user to capture the layer following certain standards.
- Efficiency: Could it be more efficient? Probably. However, this project serves as a kickstart to get bigger projects running!