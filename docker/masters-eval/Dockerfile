FROM tensorflow/tensorflow:2.4.3-gpu
LABEL maintainer="benjamin.wilhelm@uni-konstanz.de"
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libopenblas-dev
RUN pip install \
	imageio \
	pandas \
	sklearn \
	cython \
	scipy \
	tqdm \
	opencv-python \
	PyYAML \
	image-similarity-measures==0.3.5
RUN pip install \
	tensorflow-datasets==4.4.0 \
	tensorflow-probability==0.12.2
RUN pip install \
	torch==1.8.2+cpu \
	torchvision==0.9.2+cpu \
	torchaudio==0.8.2 \
	-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
RUN pip install \
	basicsr==1.3.4.2
RUN pip install \
	git+https://github.com/pathak22/pyflow@8f8ab9e90845f1b819b3833f995f481cb5d92166 \
	git+https://github.com/HedgehogCode/pyinverseflow@0.1.0 \
	git+https://github.com/HedgehogCode/tensorflow-datasets-bw.git@0.11.1 \
	git+https://github.com/HedgehogCode/deep-plug-and-play-prior@thesis
COPY eval_utils/ /eval_utils
COPY scripts/ /scripts
COPY methods/ /methods
