# BasicSR (EDSR, ESRGAN, EDVR) - SISR, VSR, LFSR

Implementation of super-resolution using BasicSR.
The EDSR, ESRGAN, and EDVR models are implemented for all applicable tasks.
Note, that for these models the official weights provided by the paper authors were used.

Download the "official" pretrained models from [BasicSR GoogleDrive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) (See [here](https://github.com/xinntao/BasicSR/blob/master/docs/ModelZoo.md)).
Place them in `methods/basicsr/pretrained_models/<model_name>/`

**Code**: https://github.com/xinntao/BasicSR

**Papers**:

- **EDSR:** B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, [“Enhanced Deep Residual Networks for Single Image Super-Resolution,”](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html) in 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Honolulu, HI, USA, Jul. 2017, pp. 1132–1140. doi: 10.1109/CVPRW.2017.151.
- **ESRGAN:** X. Wang et al., [“ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks,”](https://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html) in Computer Vision – ECCV 2018 Workshops, vol. 11133, L. Leal-Taixé and S. Roth, Eds. Cham: Springer International Publishing, 2019, pp. 63–79. doi: 10.1007/978-3-030-11021-5_5.
- **EDVR:** X. Wang, K. C. K. Chan, K. Yu, C. Dong, and C. C. Loy, [“EDVR: Video Restoration With Enhanced Deformable Convolutional Networks,”](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.html) in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Long Beach, CA, USA, Jun. 2019, pp. 1954–1963. doi: 10.1109/CVPRW.2019.00247.
