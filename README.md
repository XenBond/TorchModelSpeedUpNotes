# TorchModelSpeedUpNotes
Notes to record pytorch model compression and acceleration methods and notes

# Docker container
Steps for running in docker container:

1. build a docker image by running:
```
docker build -t demo_image .
```
2. run the image as a container
```
docker run -it --gpus "device=0" -v <current path>:/usr/src/app demo_image
```
3. step 2 should automatically let you enter the container, install the required python packages
```
pip install -r requirements.txt
```
4. run the inference and evaluation through main.py
```
python main.py
```
