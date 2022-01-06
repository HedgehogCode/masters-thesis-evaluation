# Master's Thesis Evaluation

This repository contains code to reproduce the results of my Master's Thesis.

## Getting Started

Clone the repository with submodules:

```
$ git clone --recurse-submodules https://github.com/HedgehogCode/masters-thesis-evaluation.git
```

Use conda to create the required Python environment and activate it:

```
$ conda env create -f environment.yml
$ conda activate masters-thesis-eval
```

## Notebooks

The folder [`notebooks/`](notebooks/) contains Jupyter notebooks.

- `notebooks/eval-*.ipynb`: Load the results of the respecive task and show them in a table (interactive).
- `notebooks/table-*.ipynb`: Load the results of the respecive task create the tables included in the thesis.
- `notebooks/figure-prior-noise.ipynb`: Create the figure about DMSP nb-deblurring with different levels of prior noise from the thesis.

## Model Converters

The folder [`model_converters/`](model-converters/) contains scripts for converting existing pretrained models to TensorFlow hdf5 models.

## Scripts

- `scripts/eval_*.py`: Evaluate the task on all datasets and settings.

  ```
  usage: eval_*.py [-h] [--test-run] result method [method_args]

  positional arguments:
  result       Path to the file where results will be written to.
  method       Module name of the method implementation (e.g. "methods.dmsp" or "methods.basicsr.edsr").
  method_args  Arguments for the method implementation (one string).

  optional arguments:
  -h, --help   show this help message and exit
  --test-run   Test run: Run each task only for a few steps.
  ```

- `scripts/visualize_*.py`: Visualize the method on a few examples.

  ```
  usage: visualize_denoising.py [-h] [--test-run] result method [method_args]

  positional arguments:
  result       Path to the folder where the examples will be written to.
  method       Module name of the method implementation (e.g. "methods.dmsp" or "methods.basicsr.edsr").
  method_args  Arguments for the method implementation (one string).

  optional arguments:
  -h, --help   show this help message and exit
  --test-run   Test run: Run each task only for a few steps.
  ```

- [`scripts/run_per_model.py`](scripts/run_per_model.py): Run the given script once for each model in the given direcotry.

  ```
  usage: find_prior_noise_for_dmsp.py [only_noise_range] model_dir script_path [script_args]...

  arguments:
  only_noise_range    If present: Run the script only for models with a noise range.
  model_dir           Path to the directory containing the models.
  script_path         Path to the script to run.
  script_args         Arguments for the script to run. The markers "{mn}" and "{mp}" are replaced
                      by the model name and model path, respectively.
  ```

- [`scripts/find_prior_noise_for_dmsp.py`](scripts/find_prior_noise_for_dmsp.py): Try DMSP nb-deblurring with different levels of prior noise.

  ```
  usage: find_prior_noise_for_dmsp.py [-h] [--test-run] model result

  positional arguments:
  model       Path to the model file.
  result      Path to the folder where the results will be written to.

  optional arguments:
  -h, --help  show this help message and exit
  --test-run  Test run: Run each task only for 5 step.
  ```

- [`scripts/eval_lfsr_disparity_estimation.py`](scripts/eval_lfsr_disparity_estimation.py): Try light field super-resolution with GT disparity and estimated disparity.

  ```
  usage: eval_lfsr_disparity_estimation.py [-h] [--test-run] model result

  positional arguments:
  model       Path to the model file.
  result      Path to the folder where the results will be written to.

  optional arguments:
  -h, --help  show this help message and exit
  --test-run  Test run: Run each task only for 5 step.
  ```

- [`scripts/make_figures_gt_degraded.py`](scripts/make_figures_gt_degraded.py): Create figures for GT and degraded examples.

  ```
  usage: make_figures_gt_degraded.py [-h] [--denoising DENOISING]
                                 [--nb-deblurring NB_DEBLURRING]
                                 [--sisr SISR] [--vsr VSR] [--lfsr LFSR]
                                 [--inpainting INPAINTING]

  optional arguments:
  -h, --help            show this help message and exit
  --denoising DENOISING
                          Path to the denoising figures.
  --nb-deblurring NB_DEBLURRING
                          Path to the nb-deblurring figures.
  --sisr SISR           Path to the sisr figures.
  --vsr VSR             Path to the vsr figures.
  --lfsr LFSR           Path to the lfsr figures.
  --inpainting INPAINTING
                          Path to the inpainting figures.
  ```

## Reproducing Evaluation

The results from all scripts can be downloaded at [Google Drive/results.zip](https://drive.google.com/file/d/1NrfyUeaMXBt5aOx1DxV3HYZGqk-OARGp/view?usp=sharing) and [Google Drive/figures.zip](https://drive.google.com/file/d/1buybo7_nboL0sIHn76gSsnPZHe7E7SEV/view?usp=sharing).

**Denoising**

Evaluation

```
$ python scripts/run_per_model.py models scripts/eval_denoising.py results/denoising/{mn}.csv methods.dngan {mp}
$ python scripts/eval_denoising.py results/denoising/cbm3d.csv methods.cbm3d
$ python scripts/eval_denoising.py results/denoising/dncnn.csv methods.dncnn
```

Figures

```
$ python scripts/make_figures_gt_degraded.py --denoising figures/denoising
$ python scripts/visualize_denoising.py figures/denoising methods.dngan models/drunet+_0.0-0.2.h5
$ python scripts/visualize_denoising.py figures/denoising methods.dngan models/drugan+_0.0-0.2.h5
```

**Non-Blind Deblurring**

```
$ python scripts/run_per_model.py models scripts/eval_nb_deblurring.py results/nb_deblurring/dmsp/{mn}.csv methods.dmsp {mp},nb
$ python scripts/run_per_model.py models scripts/eval_nb_deblurring.py results/nb_deblurring/dmsp_na/{mn}.csv methods.dmsp {mp},na
$ python scripts/run_per_model.py models scripts/eval_nb_deblurring.py results/nb_deblurring/hqs/{mn}.csv methods.hqs {mp}
$ python scripts/eval_nb_deblurring.py results/nb_deblurring/dmsp_paper.csv methods.dmsp_paper
$ python scripts/eval_nb_deblurring.py results/nb_deblurring/dpir.csv methods.dpir
$ python scripts/eval_nb_deblurring.py results/nb_deblurring/fdn.csv methods.fdn
```

Figures

```
$ python scripts/make_figures_gt_degraded.py --nb-deblurring figures/nb-deblurring
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.dmsp models/drunet+_0.0-0.2.h5,nb
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.dmsp models/drugan+_0.0-0.2.h5,nb
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.hqs models/drunet+_0.0-0.2.h5,nb
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.hqs models/drugan+_0.0-0.2.h5,nb
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.dpir
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.dmsp_paper
$ python scripts/visualize_nb_deblurring.py figures/nb-deblurring/ methods.fdn
```

**Single Image Super-Resolution**

```
$ python scripts/eval_sisr.py results/sisr/bicubic.csv methods.bicubic
$ python scripts/run_per_model.py models scripts/eval_sisr.py results/sisr/dmsp/{mn}.csv methods.dmsp {mp}
$ python scripts/run_per_model.py only_noise_range models scripts/eval_sisr.py results/sisr/hqs/{mn}.csv methods.hqs {mp}
$ python scripts/eval_sisr.py results/sisr/dmsp_paper.csv methods.dmsp_paper
$ python scripts/eval_sisr.py results/sisr/dpir.csv methods.dpir
$ python scripts/eval_sisr.py results/sisr/edsr.csv methods.basicsr.edsr
$ python scripts/eval_sisr.py results/sisr/esrgan.csv methods.basicsr.esrgan
```

Figures

```
$ python scripts/make_figures_gt_degraded.py --sisr figures/sisr
$ python scripts/visualize_sisr.py figures/sisr/ methods.dmsp models/drunet+_0.0-0.2.h5
$ python scripts/visualize_sisr.py figures/sisr/ methods.dmsp models/drugan+_0.0-0.2.h5
$ python scripts/visualize_sisr.py figures/sisr/ methods.hqs models/drunet+_0.0-0.2.h5
$ python scripts/visualize_sisr.py figures/sisr/ methods.hqs models/drugan+_0.0-0.2.h5
$ python scripts/visualize_sisr.py figures/sisr/ methods.bicubic
$ python scripts/visualize_sisr.py figures/sisr/ methods.dpir
$ python scripts/visualize_sisr.py figures/sisr/ methods.basicsr.edsr
$ python scripts/visualize_sisr.py figures/sisr/ methods.basicsr.esrgan
```

**Video Super-Resolution**

```
$ python scripts/eval_vsr.py results/vsr/bicubic.csv methods.bicubic
$ python scripts/run_per_model.py only_noise_range models scripts/eval_vsr.py results/vsr/hqs/{mn}.csv methods.hqs {mp}
$ python scripts/eval_vsr.py results/vsr/edsr.csv methods.basicsr.edsr
$ python scripts/eval_vsr.py results/vsr/esrgan.csv methods.basicsr.esrgan
$ python scripts/eval_vsr.py results/vsr/edvr.csv methods.basicsr.edvr
```

Figures

```
$ python scripts/make_figures_gt_degraded.py --vsr figures/vsr
$ python scripts/visualize_vsr.py figures/vsr/ methods.hqs models/drunet+_0.0-0.2.h5
$ python scripts/visualize_vsr.py figures/vsr/ methods.hqs models/drugan+_0.0-0.2.h5
$ python scripts/visualize_vsr.py figures/vsr/ methods.bicubic
$ python scripts/visualize_vsr.py figures/vsr/ methods.basicsr.edsr
$ python scripts/visualize_vsr.py figures/vsr/ methods.basicsr.esrgan
$ python scripts/visualize_vsr.py figures/vsr/ methods.basicsr.edvr
```

**Light Field Super-Resolution**

```
$ python scripts/eval_lfsr.py results/lfsr/bicubic.csv methods.bicubic
$ python scripts/run_per_model.py only_noise_range models scripts/eval_lfsr.py results/lfsr/hqs/{mn}.csv methods.hqs {mp}
$ python scripts/eval_lfsr.py results/lfsr/edsr.csv methods.basicsr.edsr
$ python scripts/eval_lfsr.py results/lfsr/esrgan.csv methods.basicsr.esrgan
$ python scripts/eval_lfsr.py results/lfsr/edvr.csv methods.basicsr.edvr
$ python scripts/eval_lfsr.py results/lfsr/lfssr.csv methods.lfssr
```

Figures

```
$ python scripts/make_figures_gt_degraded.py --lfsr figures/lfsr
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.hqs models/drunet+_0.0-0.2.h5
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.hqs models/drugan+_0.0-0.2.h5
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.bicubic
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.basicsr.edsr
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.basicsr.esrgan
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.basicsr.edvr
$ python scripts/visualize_lfsr.py figures/lfsr/ methods.lfssr
```

**Inpainting**

Figures

```
$ python scripts/make_figures_gt_degraded.py --inpainting figures/inpainting
$ python scripts/visualize_inpainting.py figures/inpainting/ methods.dmsp models/drunet+_0.0-0.2.h5
$ python scripts/visualize_inpainting.py figures/inpainting/ methods.dmsp models/drugan+_0.0-0.2.h5
$ python scripts/visualize_inpainting.py figures/inpainting/ methods.hqs models/drunet+_0.0-0.2.h5
$ python scripts/visualize_inpainting.py figures/inpainting/ methods.hqs models/drugan+_0.0-0.2.h5
```
