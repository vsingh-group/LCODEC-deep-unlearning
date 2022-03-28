To run basic scrubbing, check scripts in [scrub_scripts](https://github.com/ronakrm/lcodec_unlearning/tree/main/scrub/scrub_scripts) folder.

Create results/ and trained_models/ subfolders for output files and models.

To run training and scrubbing from different settings (MNIST, CIFAR-10) use the following scripts:

```
bash scrub_scripts/mnist_logistic.sh 
```
for scrubbing a logistic regression model on MNIST

```
bash scrub_scripts/mnist_2nn.sh
```
for scrubbing a two layer neural network model on MNIST

```
bash scrub_scripts/cifar_scrub.sh
```
for scrubbing several deep neural networks on CIFAR-10

For scrubbing from DistilBERT based transformer on LEDGAR dataset (mentioned in paper) use
```
python ledgar_scrub.py --data <Path to .jsonl file containing dataset> --mode train
```
for fine-tuning the transformer model

and

```
python ledgar_scrub.py --data <Path to .jsonl file containing dataset> --mode scrub
```
for scrubbing from the above fine-tuned transformer model

Note: The main file for multiple removals from a trained model is [multi_scrub.py](https://github.com/ronakrm/lcodec_unlearning/blob/main/scrub/multi_scrub.py)

For VGG-Face identification, use
```
bash scrub_scripts/vgg_scrub.sh
```
data not included in upload or in scripts due to size and nature of images and licenses. See dl_imgs.py in data/vggface directory.
Data downloading is very network-intensive and connects to many servers, use care when generating dataset.


For Person Re-Identification:

Please install the modified source of [torch-reid](https://github.com/KaiyangZhou/deep-person-reid) as provided in the folder deep-person-reid-master. We have modified the engine to make it compatible to our scrubbing code. The main file for deep unlearning of person re-identification models can be run using
```
python reid_scrub.py
```

For Retraining experiments, one can use:
```
bash scrub_scripts/mnist_retrain_new.sh
```
Note: There are other retraining scripts in the [scrub_scripts](https://github.com/ronakrm/lcodec_unlearning/tree/main/scrub/scrub_scripts) folder. The main file for retraining is [retrain_scrub.py](https://github.com/ronakrm/lcodec_unlearning/blob/main/scrub/retrain_scrub.py)


For Running KFAC based Hessian approximation for unlearning:
```
python kfac_scrub.py --do_train --do_KFAC
```
This will train and then unlearn a multi layer fully connected neural network on the MNIST dataset. In order to compare with LCODEC based scrubbing one can remove the use of --do_KFAC flag and change the output files accordingly.

KFAC based scrubbing is derived from the Pytorch implementation of KFAC found [here](https://github.com/cybertronai/autograd-lib)
