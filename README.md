# segmentation-of-excavator-teeth

The purpose of the work: to solve the problem of Instance Segmentation in order to detect potentially dangerous scenarios for the loss of bucket teeth and to justify the introduction of one of them into the excavator teeth control system developed by the Ciphra technology company. 
Within the framework of this work, modern approaches of solving the Instance Segmentation problem are first analyzed, a data markup technique is developed, data markup is performed, an easily reproducible development environment is created inside the container using Docker and the Mask-RCNN deep learning neural network is trained on the marked data as a baseline solution. It is also possible to deploy architectures such as SOLOv2, BlendMask and SondInst for Instance Segmentation tasks inside this environment.

 Its my representation of adelaidet and detectron2 repos for instance segmentation task. 
 By defalt Mask R_SNN is set as a backbone. Default trainer achieves mAP 37.5 for segmentations and mAP 47.4 for bboxes with 60000 iterations.
 Original code of adet and detectron2 is presented below: 
 1. https://github.com/aim-uofa/AdelaiDet
 2. https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html


# How to build with Docker: 

### use presented dockerfile to build an image
sudo docker build -t seg-app:2.0 .
### run the image in contaner specifying directory with COCO-format dataset on your host
sudo docker run -it --gpus all --shm-size=30gb -v "--your_dir_with_coco_format_dataset--:/home" seg-app:2.0

### copy mounted dataset to the work directory
cp -R /home/.../coco   /root/code/adet/datasets 

### create virtualenv to show detectron2 data directory
export DETECTRON2_DATASETS=/root/code/adet/datasets

### instal cv2
pip install opencv-python-headless

### run to train model and eval. To change training params change the file
python train_and_eval.py  

### inference trained model on test data. By default uses model_final.pth
python inference_and_eval.py
![Screenshot from 2022-05-16 16-01-55](https://user-images.githubusercontent.com/57952207/168669872-34f4b95a-28ac-49bc-a0f5-5ba6e6399db7.png)
