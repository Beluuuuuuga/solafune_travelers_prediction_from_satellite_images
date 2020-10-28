FROM tensorflow/tensorflow:2.3.1-gpu-jupyter

# RUN sudo yum install -y mesa-libGL.x86_64
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip install pandas
RUN pip install tqdm
RUN pip install opencv-python
RUN pip install tensorflow-addons
RUN pip install scikit-learn
RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/mjkvaak/ImageDataAugmentor.git@ce9bf4d7d532bfcb14fda7fb43d7bcdc6d7990ff