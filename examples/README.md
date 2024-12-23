# **Examples**  

This repository includes various examples of running ML models from different application areas using **Tritonserver-rs**. Additionally, you’ll find:  
- Instructions on how to run models in different formats.  
- Steps to configure a CPU-only setup.  
- Guides to convert `.pth` models into `.pt`, `.onnx`, or `.trt`.  
- Details on how to develop and deploy applications that use Tritonserver-rs.  

All models included are sourced from the public domain and selected nearly at random to demonstrate Tritonserver-rs's ability to handle any ML model **without requiring additional code modifications**.  

Before starting a pipeline, the corresponding model must be downloaded (except for the "simple" pipeline). Documentation about where to find the models and how to prepare them for Triton is available in the [models folder](./models/README.md).  

---

## **How to Run Pipelines**  

The examples are designed to run inside Docker containers for ease of deployment.  

### **Step 1: Build the Development Image**  
If you haven’t already built the development Docker image, follow the steps in [DEPLOY.md](../DEPLOY.md):  
```sh
$ docker build --network host -t triton_dev:24.07 --build-arg TRITON_CONTAINER_VERSION=24.07 -f ../Dockerfile.dev .
```

### **Step 2: Build the Examples Image**  
Once the development image is ready, build the image for running the pipelines. This image includes all the necessary libraries for the models:  
```sh
$ docker build --network host -t examples_img:0.1 --build-arg DEV_IMAGE=triton_dev:24.07 -f ./Dockerfile .
```

### **Step 3: Run the Container**  
#### Interactive Mode:  
To make code, data, or model edits, you can start the container in interactive mode:  
```sh
$ docker run -it --rm --entrypoint sh -v .:/opt/tritonserver/workspace --network=host examples_img:0.1
```  

#### Non-Interactive Mode:  
If no edits are needed, simply run the container:  
```sh
$ docker run -it --rm --network=host -v .:/opt/tritonserver/workspace examples_img:0.1 \
    "binary name" [binary CLI args]
```

---

## **Pipelines Overview**  

### **1. Image Detection**  
Draws bounding boxes, class labels, and confidence scores for objects detected using a YOLO model. This pipeline uses multiple models: an ONNX base model for detection and a Python model for postprocessing.  

**Command:**  
```sh
$ docker run -it --rm --network=host --shm-size=128Mb \
    -v .:/opt/tritonserver/workspace examples_img:0.1 \
    "image_detection" --image_path=data/input/traffic.jpg --output_path=data/output/traffic_bb.jpg
```  

**Result:**  
<p float="left">
  <img src="./data/input/traffic.jpg" alt="Before" style="width:600px; height:450px;"/>
  <img src="./data/output/traffic_bb.jpg" alt="After" style="width:600px; height:450px;"/>
</p>  

---

### **2. Simple Model**  
A minimal example of creating a custom model. The model sums up all tensor values. Written in Python. It also serves as custom tracing and allocation example.

**Command:**  
```sh
$ docker run -it --rm --network=host -v .:/opt/tritonserver/workspace examples_img:0.1 \
    "simple" -i=[1,2,3,36]
```  

**Result:**  
```sh
[INFO  simple] Sum by dims of [1.0, 2.0, 3.0, 36.0] is: 42
```

---

### **3. Audio Classification**  
Detects and classifies audio events in a recording, logging the two most likely classes for each frame. Uses a TensorFlow model (`yamnet`).  

**Command:**  
```sh
$ docker run -it --rm --network=host -v .:/opt/tritonserver/workspace examples_img:0.1 \
    "audio" -i=data/input/fight-club-trailer.wav
```  

**Result:**  
```sh
[INFO  audio] Max scores for each frame: [("Music", "Scary music"), ("Speech", "Music"), ("Music", "Singing"), ...]
```

---

### **4. Image Generator**  
Generates random handwritten-like images based on the MNIST dataset, arranging them into an NxN grid and saving as a PNG. Uses a PyTorch model.  

**Command:**  
```sh
$ docker run -it --rm --network=host -v .:/opt/tritonserver/workspace examples_img:0.1 \
    "generator" --table_size=7 -o=data/output/mnist.png
```  

**Result:**  
<img src="./data/output/mnist.png" alt="Result image"/>  

---

### **5. Text Detection**  
Detects text in an image and draws bounding boxes around each piece of text. Supports PyTorch, ONNX, and TensorRT models.  

**Command:**  
```sh
$ docker run -it --rm --network=host -v .:/opt/tritonserver/workspace examples_img:0.1 \
    "text_detection" -i=data/input/traffic.jpg --output_path=data/output/traffic_bb_text.jpg
```  

**Result:**  

**Note:** Documentation for this pipeline is still in progress.  

--- 

This README provides a quick guide to set up and run pipelines using Tritonserver-rs. Explore each example to understand its functionality and how it demonstrates the versatility of Tritonserver-rs!  