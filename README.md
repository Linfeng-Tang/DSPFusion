# This is official Pytorch implementation of "DSPFusion: Degradation and Semantic Prior Dual-guided Framework for Image Fusion"

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
