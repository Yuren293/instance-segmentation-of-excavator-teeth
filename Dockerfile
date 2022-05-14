FROM nvidia/cuda:11.6.0-devel-ubuntu18.04

RUN apt-get update && apt-get install -y libglib2.0-0 && apt-get clean

RUN apt-get install -y wget htop byobu git gcc g++ vim libsm6 libxext6 libxrender-dev lsb-core

#RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh


#RUN cd /root && bash Anaconda3-2020.07-Linux-x86_64.sh -b -p ./anaconda3
RUN cd /root && bash Anaconda3-2021.11-Linux-x86_64.sh -b -p ./anaconda3

RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda install -y pytorch==1.10.1 torchvision==0.11.2  cudatoolkit=11.3 -c pytorch -c conda-forge"

RUN bash -c "/root/anaconda3/bin/conda init bash"

WORKDIR /root
RUN mkdir code
WORKDIR code

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda activate base && cd detectron2 && python setup.py build develop"

RUN git clone https://github.com/aim-uofa/AdelaiDet.git adet

WORKDIR adet
RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda activate base && python setup.py build develop"

RUN rm /root/Anaconda3-2021.11-Linux-x86_64.sh

RUN wget https://raw.githubusercontent.com/Yuren293/instance-segmentation-of-excavator-teeth/main/train_and_eval.py

RUN wget https://raw.githubusercontent.com/Yuren293/instance-segmentation-of-excavator-teeth/main/inference_and_eval.py

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zUq4uyQB1yK_CJDOuui7LWuC7J0wbx88' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zUq4uyQB1yK_CJDOuui7LWuC7J0wbx88" -O model_0009999.pth && rm -rf /tmp/cookies.txt

RUN mkdir output

RUN cp -R /root/code/adet/model_0009999.pth /root/code/adet/output/ 
 
RUN export DETECTRON2_DATASETS=/root/code/adet/datasets 

RUN apt install nano



 
