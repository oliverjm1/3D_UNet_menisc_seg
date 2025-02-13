# Training 3D U-Net and Segment Anything Model (SAM) on MRI Knee Images for Meniscus Segmentation

Repo containing code where SAM was fine-tuned on IWOAI 2019 knee MRI data.
A 3D U-Net was also trained as a baseline.

Models, dataset classes, metrics and utility functions are all defined in `src`.

Train scripts for models are found in `scripts` folder.

`notebooks` contains all jupypter notebooks, most of which just test the code contained in the train scripts.
Other notebooks are for extracting results from the generated masks (`Bland_Altman_Plots.ipynb`, `Dice_scores.ipynb`, `Hausdorff_Distance.ipynb`)

`data` folder is empty. Put data here after cloning.

Directory Tree (OUT OF DATE):
```bash
.
├── LICENSE
├── README.md
├── data
│   └── data.md
├── knees.yml
├── models
│   └── models.md
├── notebooks
│   ├── Bland_Altman_Plots.ipynb
│   ├── Dice_scores.ipynb
│   ├── Hausdorff_Distance.ipynb
│   ├── convert_to_slices.ipynb
│   ├── run_test_split_through_sam.ipynb
│   ├── test_sam.ipynb
│   ├── test_sam_slice_files.ipynb
│   └── test_unet.ipynb
├── scripts
│   ├── hyperparams_sam.txt
│   ├── hyperparams_unet.txt
│   ├── train_SAM.py
│   ├── train_SAM_slices.py
│   └── train_UNet.py
└── src
    ├── datasets.py
    ├── metrics.py
    ├── model_SAM.py
    ├── model_UNet.py
    └── utils.py
```
