FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install albumentations==1.3.1 \
    opencv-python==4.12.0.88 \
    pandas==2.3.1 \
    scikit-learn==1.7.0 \
    matplotlib==3.10.3 \
    shapely==2.1.1 \
    tqdm prettytable rich yapf wandb
RUN pip3 install -U openmim

WORKDIR /workspace

COPY mmcv /workspace/mmcv
COPY mmdetection /workspace/mmdetection
# COPY ../mmyolo /workspace/mmyolo
COPY . /workspace/mmyolo

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

ENV TORCH_CUDA_ARCH_LIST="9.0"

RUN if [ ! -f /usr/local/cuda/lib64/libcudart.so ]; then \
      ln -s /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so; \
    fi && \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig

RUN find /workspace/mmcv -name '_ext*.so' -delete && \
    find /workspace/mmcv -name '*.cpython-*.so' -delete && \
    rm -rf /workspace/mmcv/build

RUN cd /workspace/mmcv && pip install -v -e .
RUN cd /workspace/mmdetection && pip install -v -e .
RUN cd /workspace/mmyolo && pip install -r requirements/albu.txt && mim install -v -e .
# RUN cd /workspace/mmyolo && mim download mmyolo --config yolov5_s-v61_syncbn_fast_8xb16-300e_coco --dest .
