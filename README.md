# LiftReg: Limited Angle 2D/3D Deformable Registration

This is the official repository for

**LiftReg: Limited Angle 2D/3D Deformable Registration.** \
[Lin Tian](https://www.cs.unc.edu/~lintian/), Yueh Z. Lee, Ra\'ul San Jos\'e Est\'epar, Marc Niethammer\
MICCAI 2022. [Paper](https://arxiv.org/abs/2203.05565)

In this work we propose LiftReg, a 2D/3D deformable registration approach. LiftReg is a deep registration framework 
which is trained using sets of digitally reconstructed radiographs (DRR) and computed tomography (CT) image pairs. By using simulated training data, LiftReg can use a high-quality CT-CT image similarity measure, which helps the network to learn a high-quality deformation space. To further improve registration quality and to address the inherent depth ambiguities of very limited angle acquisitions, we propose to use features extracted from the backprojected 2D images and a statistical deformation model. We test our approach on the DirLab lung registration dataset and show that it outperforms an existing learning-based pairwise registration approach. 
![Model Structure](/readme_materials/NetworkDiagram.png)




## Setup environment
```
cd LiftReg
conda create -n liftreg python=3.7
pip install -r requirements.txt
```

## Train with built subspace
Assume the two files (pca_vectors.npy and pca_mean.npy) are saved in PCA_PATH and the path to the dataset folder is DATA_PATH.
1. Set the "pca_path" in cur_task_setting.json to PCA_PATH
2. Run the following commands
```
cd LiftReg
python main.py -o OUTPUT_FOLDER -d DATA_PATH -e disp_subspace -s ./ -g GPU_ID
```

## Evaluate with the trained model
If the model is trained following the commands in the previous section, skip step 1. The paths are set correctly by main.py script and the setting.json file is saved at the OUTPUT_FOLDER/disp_subspace/DATE/ folder.

### Step 1. Set the paths in cur_task_setting.json
To run the code with pretrained model, set the following paths in cur_task_setting.json:
1. ["dataset"]["data_path"]
2. ["train"]["output_path"]

### Step 2. Run evaluation script
Run the following command:
```
cd LiftReg
python eval.py -s ./exps/disp_subspace/2022_02_17_20_12_57/cur_task_setting.json -g GPU_ID
```

## Pre-built subspace
The link to the built subspace for lung motion is coming soon.

## Citation
```
@inproceedings{tian2022liftreg,
  title={LiftReg: Limited Angle 2D/3D Deformable Registration},
  author={Tian, Lin and Lee, Yueh Z and San Jos{\'e} Est{\'e}par, Ra{\'u}l and Niethammer, Marc},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={207--216},
  year={2022},
  organization={Springer}
}

```
