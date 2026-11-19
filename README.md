
<h1>Performance estimation inference</h1>
<p>This is an attempt to deploy the performance prediction on real life videos</p>
## Prerequisites and Requirements
<!-- NOTE: - two environments, one for all models without Swin-T, and one for Swin-T based models -->

For more direct reproducibility from the paper we recommend using separate environments: One for non-Transformer-based models (ResNet, ConvNext-T), and the other one for Transformer-based models (Swin-T).

<!-- To install the environment from scratch, follow the following steps:
```
conda create --name swiftnet-pp-v2 python=3.7.12
source activate swiftnet-pp-v2
conda install pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=10.2 -c pytorch
conda install matplotlib=3.2.2 scipy=1.7.3 scikit-image=0.19.3 numba=0.55.1
conda install -c conda-forge wand=0.6.5
pip install opencv-python==4.5.5.62
pip install "git+https://github.com/ifnspaml/IFN_Dataloader.git"
pip install "git+https://github.com/ifnspaml/TUBSRobustCheck.git"
``` -->

## Installation
To install the environment via environment.yml, follow the following steps:
```
conda env create --file environment.yml
source activate swiftnet-pp-v2
pip install "git+https://github.com/ifnspaml/IFN_Dataloader.git"
pip install "git+https://github.com/ifnspaml/TUBSRobustCheck.git"
```
Analog to above installation for training the Transformer-based models, follow:
```
conda env create --file environment_tf.yml
source activate swiftnet-pp-v2-tf
pip install "git+https://github.com/ifnspaml/IFN_Dataloader.git"
pip install "git+https://github.com/ifnspaml/TUBSRobustCheck.git"
```
<h2>TODOs:</h2>
<p>
we can get the PSNR but what do we need to estimate the mIoU?

some ideas:
<ul>
    <li>
        get the regression coefficient and just calculate the estimated mIoU from that
    </li>
    <li>
        using bayes inference or kalman filter
    </li>
</ul>

bayes inference or kalman filter:
<ul>
    <li>
        transform the PSNR and the mIoU from the val data into a probability density. we need:
        <ul>
            <li>transition matrix</li>
            <li>state matrix</li>
            <li>noise covariance matrix</li>
        </ul>
    </li>
</ul>
</p>