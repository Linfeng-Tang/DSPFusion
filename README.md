# DSPFusion: Image Fusion via Degradation and Semantic Dual-Prior Guidance

Official PyTorch implementation of **DSPFusion**, accepted as a regular paper by **IEEE Transactions on Image Processing (IEEE TIP)** on June 2, 2026.

**Authors:** Linfeng Tang, Chunyu Li, Yeda Wang, Guoqing Wang, Yixuan Yuan, and Jiayi Ma

[[IEEE Xplore / DOI](https://doi.org/10.1109/TIP.2026.3700938)] [[arXiv](https://arxiv.org/abs/2503.23355)]

## Fast Testing

1. **Create a virtual environment for DSPFusion**
   ```
   conda create -n DSPFusion python=3.9
   pip install -r requirements.txt
   ```
2. **Run the following code for faster testing:**
3. ```
   python test.py -opt=./options/test/DSPF_S2.yml
   ```

## Fast Training

1. **Build the training dataset as shown in “. /datasets/Hybrid_Datasets.**

   ```
     Hybrid_Datasets
     |——train
         |——IR
             |——image1.png
             |——image2.png
             ......
         |——IR_enhanced
             |——image1.png
             |——image2.png
             ......
         |——VI
             |——image1.png
             |——image2.png
             ......
         |——VI_enhanced
             |——image1.png
             |——image2.png
             ......
     |——val
         |——IR
             |——image1.png
             |——image2.png
             ......
         |——IR_enhanced
             |——image1.png
             |——image2.png
             ......
         |——VI
             |——image1.png
             |——image2.png
             ......
         |——VI_enhanced
             |——image1.png
             |——image2.png
             ......
   ```
2. **Run the following code for Stage I training:**

   ```
   python train.py -opt=./options/train/DSPF_S1.yml
   ```
3. **Run the following code for Stage II training:**

   ```
   python train.py -opt=./options/train/DSPF_S1.yml
   ```

## Citation

```bibtex
@article{Tang2026DSPFusion,
  title={DSPFusion: Image Fusion via Degradation and Semantic Dual-Prior Guidance},
  author={Tang, Linfeng and Li, Chunyu and Wang, Yeda and Wang, Guoqing and Yuan, Yixuan and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  doi={10.1109/TIP.2026.3700938}
}
```
