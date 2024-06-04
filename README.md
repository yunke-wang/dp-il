## Imitation Learning from Purified Demonstrations

This repository contains the PyTorch code for the paper "Imitation Learning from Purified Demonstrations" in ICML 2024.

## Requirements
Experiments were run with Python 3.6 and these packages:
* pytorch == 1.1.0
* gym == 0.15.7
* mujoco-py == 2.0.2.9

## Train

 * Train Diffusion Models with Few Optimal Demonstrations
 ```
  python ddpm_il.py --env_id 1/2/3/4 --il_method diffusion --action diff
 ```

* Behavior Cloning with Purified Demonstrations
 ``` 
  python ddpm_il.py --env_id 1/2/3/4 --c_data 1/2/3/4 --il_method diffusion --action diff --diff_t 5/10/30/50/100 --noise_level 1/2/3
 ```

* GAIL with Purified Demonstrations
 ``` 
  python ddpm_il.py --env_id 1/2/3/4 --c_data 1 --il_method diffusion --action gail --denoise --diff_t 5 --noise_level 1/2/3  --seed 0/1/2/3/4
  ```

 
The re-implementation of BCND/DWBC/WGAIL/2IWIL/IC-GAIL/WGAIL can be found in core/irl.py.

## Contact

For any questions, please feel free to contact me at yunke.wang@whu.edu.cn.

## Citation
```
@inproceedings{wang2024imitation,
  title={Imitation Learning from Purified Demonstrations},
  author={Wang, Yunke and Dong, Minjing and Zhao, Yukun and Du, Bo and Xu, Chang},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Acknowledgement
We thank the authors of [VILD](https://github.com/voot-t/vild_code). Our code structure is based on their source code and we also use some of expert data collected by VILD.

## Reference
[1] Generative adversarial imitation learning. NeurIPS 2016.

[2] Learning robust rewards with adversarial inverse reinforcement learning. ICLR 2018.

[3] Variational discriminator bottleneck: Improving imitation learning, inverse rl, and gans by constraining information flow. ICLR 2017.

[4] InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations. NeurIPS 2017

[5] Imitation learning from imperfect demonstration. ICML 2019.

[6] Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations. ICML 2019.

[7] Better-than-demonstrator imitation learning via automatically-ranked demonstrations. CoRL 2020.

[8] Variational Imitation Learning with Diverse-quality Demonstrations. ICML 2020.

[[9]](https://github.com/yunke-wang/WGAIL) Learning to Weight Imperfect Demonstrations. ICML 2021

[[10]](https://github.com/yunke-wang/SAIL) Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations. IJCAI 2021.

[[11]](https://github.com/yunke-wang/UID) Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning. AAAI 2021.
