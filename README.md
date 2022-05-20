# Deep Learning Based Diagnosis of Parkinson’s Disease Using Diffusion Magnetic Resonance Imaging
This repository is the official implementation of our Brain Imaging and Behavior 2022 paper titled 'Deep Learning Based Diagnosis of Parkinson’s Disease Using Diffusion Magnetic Resonance Imaging'. The codes are divided to two parts, training and combination, as follows:

## Training subregional CNNs
Enter the training/ directory, and run the following commands:

    CUDA_VISIBLE_DEVICES=0 python region_train.py 

for training and saving the CNN model.

    CUDA_VISIBLE_DEVICES=0 python extract_feature.py

for recording the results in **.xlsx** file format.

## Combining the trained subregion CNNs
Run the following command:

    CUDA_VISIBLE_DEVICES=0 python val_voting.py

for finding the best combination and giving the final results.

## Citation
Please cite our work if you found it useful:

    @article{zhao2022deep,
      title={Deep learning based diagnosis of Parkinson’s Disease using diffusion magnetic resonance imaging},
      author={Zhao, Hengling and Tsai, Chih-Chien and Zhou, Mingyi and Liu, Yipeng and Chen, Yao-Liang and Huang, Fan and Lin, Yu-Chun and Wang, Jiun-Jie},
      journal={Brain Imaging and Behavior},
      pages={1--12},
      year={2022},
      publisher={Springer}
    }
