# Monocular Vision-Based Traversability Estimation for Offroad Navigation 

## Abstract 

The ability to perceive and understand terrain traversability is paramount for the safe and efficient navigation 
of unstructured, off-road environments. In this work, we present a real-time system designed to generate a 
traversability map of the off-road environment using only monocular RGB images. We leverage the GOOSE dataset, 
which provides aligned RGB images and semantic masks, to train and fine-tune a DeepLabv3+ segmentation model with a 
MobileNet backbone. Our focus is on binary segmentation of traversable versus non-traversable terrain, derived 
from a curated set of semantic classes. We construct a PyTorch training pipeline with custom dataset handling, 
on-the-fly data filtering, and validation-based checkpointing. The final model is exported to ONNX and 
integrated into a real-time C++ semantic segmentation system. This pipeline enables scalable, sensor-efficient terrain 
understanding for autonomous ground vehicles (AGVs) operating in complex natural environments, 
offering a step toward low-cost, vision-based off-road navigation system.  

<div style="display: flex; justify-content: center;">
  <img src="media/bicycle_overlay.gif" alt="Overlay" width="500">
</div>
_____________________________________________________________________________________________________________________
