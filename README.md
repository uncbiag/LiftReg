# LiftReg: Limited Angle 2D/3D Deformable Registration

This is the official repository for

**LiftReg: Limited Angle 2D/3D Deformable Registration.** \
Lin Tian, Yueh Z. Lee, Ra\'ul San Jos\'e Est\'epar, Marc Niethammer
<!-- [Paper](https://drive.google.com/file/d/1-gORB0x9qa8hDpnpLSISXGmb9I6j9SG9/edit) -->

In this work we propose LiftReg, a 2D/3D deformable registration approach. LiftReg is a deep registration framework 
which is trained using sets of digitally reconstructed radiographs (DRR) and computed tomography (CT) image pairs. By using simulated training data, LiftReg can use a high-quality CT-CT image similarity measure, which helps the network to learn a high-quality deformation space. To further improve registration quality and to address the inherent depth ambiguities of very limited angle acquisitions, we propose to use features extracted from the backprojected 2D images and a statistical deformation model. We test our approach on the DirLab lung registration dataset and show that it outperforms an existing learning-based pairwise registration approach. 
![Model Structure](/readme_materials/NetworkDiagram.png)




# How to run
Setup environment
```
cd LiftReg
conda create -n liftreg python=3.7
pip install -r requirements.txt
```
To run the code with pretrained model, set the dataset folder in [here]. Then run the following command:
```
cd LiftReg
python eval.py -s ./exps/disp_subspace/2022_02_17_20_12_57/cur_task_setting.json -g [GPU_id]
```
