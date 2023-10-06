# Prex-Net
Prex-Net : Progressive Exploration Network using Efficient Channel Fusion for Light Field Reconstruction

Light field (LF) reconstruction is a technique for synthesizing views between LF images, and various methods have been proposed to obtain high quality LF reconstructed images. In this paper, we propose a progressive exploration network using efficient channel fusion for light field reconstruction (Prex-Net), which consists of three parts to fast produce high-quality synthe-sized LF images. The initial feature map extraction module uses 3D convolution to obtain deep correlations between multiple LF input images. In the channel fusion module, the extracted ini-tial feature maps pass through sequential up & down fusion blocks and continuously search for features required for LF reconstruction. Fusion block collects pixels of channels by pixel shuf-fling and applies convolution to the collected pixels to fuse the information existing between channels. Finally, the LF restoration module synthesizes LF images with high angular resolution through simple convolution using the concatenated outputs of down fusion blocks. The pro-posed Prex-Net synthesizes views between LF images faster than existing LF restoration meth-ods and shows good results in PSNR performance of the synthesized image.

## Requirements
- Python 3.9
- PyTorch 1.13

## Dataset
The test dataset can be downloaded at:

https://drive.google.com/drive/folders/1HgdVvhTBcHkH0getbNebHWaQZGX92mzc?usp=sharing
and save the test dataset files to "TestData" folder.

## Test
The trained model can be downloaded at:

https://drive.google.com/file/d/1mXp_PJKDhXPR0287FEHgPcQywti87a7U/view?usp=sharing
and save the pretrained model to "Model" folder.
