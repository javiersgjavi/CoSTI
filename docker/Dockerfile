FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN apt-get -y update
RUN apt-get -y install git
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch-scatter==2.1.1 torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.1+cu118.html
RUN pip install git+https://github.com/TorchSpatiotemporal/tsl.git