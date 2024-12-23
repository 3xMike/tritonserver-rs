ARG DEV_IMAGE

FROM ${DEV_IMAGE}
# Install the curl and build-essential packages
RUN apt-get update && \
    apt install -y python3 && \
    apt-get install -y --no-install-recommends \
    python3-pip

# Ultralytics for Yolo
RUN pip install ultralytics==8.0.51
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

# CV requirements for CRAFT
RUN pip install torch \
    torchvision \
    opencv-python \
    scikit-image \
    scipy \
    attrdict \
    yacs \
    icecream \
    onnxruntime \
    onnxruntime-gpu \
    --extra-index-url https://pypi.ngc.nvidia.com \
    nvidia-pyindex \
    onnx_graphsurgeon

RUN mkdir -m=777 /opt/tritonserver/workspace
WORKDIR /opt/tritonserver/workspace
COPY ./entrypoint.sh /opt/tritonserver/entrypoint.sh

ENTRYPOINT ["/opt/tritonserver/entrypoint.sh"]