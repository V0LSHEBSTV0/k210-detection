## Train model

### Prepare

Works only for `Linux`

* prepare python enviroment

```
pip install -r requirements.txt
```

* [Download nncase](https://github.com/kendryte/nncase/releases/tag/v0.1.0-rc5) and unzip it 
* Edit `instance/config.py` according to your hardware
* Put your dataset, in the `datasets` directory. Only supports PASCAL VOC format

### Object detection (YOLO V2)

Edit datasets_path, out_dir, dataset_dir if you need to.
Run all cells in `train.ipynb`
Make sure, that `boot.py`, `m.kmodel` appeared in out directory  

### Convert to .kmodel

On successful trainig, your model shoud be in `out/m.h5`

Open the directory in the terminal where you unziped nncase. Put in here `out` then run follow command

```
./ncc -i tflite -o k210model --dataset "out/sample_images" "out/m.tflite" "m.h5"
```

## Load model to Maix-1 k210 series 

### Prepare

* [Download ch340 driver](https://wiki.iarduino.ru/page/ch340-win-ten/) and install it
* [Download kflash_gui](https://github.com/sipeed/kflash_gui/releases/tag/v1.8.1), unzip and install it

### Loading model and script

Open kflash_gui select your `m.kmodel` and load it to 0x30000 address in flash

In root directory of your SD card put `boot.py` file.

At this point, as soon as you apply power to k210 model shoud work.