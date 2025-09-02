# Dino Visual Odometry

## Setup and Installation
The code was tested on Ubuntu 24 and Cuda 12.

Clone the repo
```
git clone https://github.com/HaktanM/DVO.git
cd DVO
```

Create a virtual environment and activate it. Then

```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install 
pip install -r requirements.txt --no-build-isolation
pip install . --no-build-isolation

# download models and data (~2GB)
./download_models_and_data.sh
```


## Evaluation


### EuRoC
Download all sequences from [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (download ASL format)
```bash
python demo.py --eurocdir=<path to EuRoC> --calib=calib/euroc.txt
```

## Training
Make sure you have run `./download_models_and_data.sh`. Your directory structure should look as follows

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

To train (log files will be written to `runs/<your name>`). Model will be run on the validation split every 10k iterations
```
python train.py --steps=240000 --lr=0.00008 --name=<your name>
```

## Acknowledgements
* The architecture is adopted from [DPVO](https://github.com/princeton-vl/DPVO)
