# segmentation-of-excavator-teeth
 Its my representation of adelaidet and detectron2 repos :
 1. https://github.com/aim-uofa/AdelaiDet
 2. https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html


# How to build with Docker: 

### use presented dockerfile to build an image
sudo docker build -t seg-app:2.0 .
### run the image in contaner specifying directory with COCO-format dataset on your host
sudo docker run -it --gpus all --shm-size=30gb -v "--your_dir_with_coco_format_dataset--:/home" seg-app:2.0

### copy mounted dataset to the work directory
cp -R /home/.../coco /root/code/adet/datasets   

pip install opencv-python-headless

### run to train model and eval. To change traing params change the file
python train_and_eval.py  

### inference trained model on test data. By default uses model_0009999.pth
python inference_and_eval.py
