# Deep Battery Saver: End-to-end Learning for Power Constrained Contrast Enhancement

![](./Demo/Demo_framework.png)

The details about our deep battery saver can be found in our [paper](https://ieeexplore.ieee.org/document/9089316).

## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this code in a scientific publication, please cite the following [paper](https://ieeexplore.ieee.org/document/9089316):

    @ARTICLE{yin_tmm20_pcce,
    author={J. -L. {Yin} and B. -H. {Chen} and Y. -T. {Peng} and C. -C. {Tsai}},
    journal={IEEE Transactions on Multimedia}, 
    title={Deep Battery Saver: End-to-End Learning for Power Constrained Contrast Enhancement}, 
    year={2021},
    volume={23},
    number={},
    pages={1049-1059},
    doi={10.1109/TMM.2020.2992962}}
    
## Dependencies
* Python 3
* Anaconda 3
* [Tensorflow >= 2.0.0](https://www.tensorflow.org/) (CUDA version >= 10.0 if installing with CUDA. [More details](https://www.tensorflow.org/install/gpu/))
* Python packages: You can install the required libraries by the command `conda DeepBatterySaver -n recreated_env --file requirements.txt`. We checked this code on cuda-10.0 and cudnn-7.6.5.

## It was tested and runs under the following OSs:
* Windows 10
* Ubuntu 16.04/18.04

Might work under others, but didn't get to test any other OSs just yet.

## Usage
1. Clone this github repo. 
```
git clone https://github.com/bigmms/deep_battery_saver
```
2. Place your testing images in `./test` folder. (There are several sample images).
3. Then, `cd` to `deep_battery_saver` and run following command for evaluation:
```
python main.py --source_path ./test/pills.jpg
```
4. To run with different settings, add `--num_epochs`, `--power_level`, `--learning_rate`, `--source_path`, `--target_path` as you need.
5. The results are in `./results` folder.

## Results

![](./Demo/Demo_results.png)
