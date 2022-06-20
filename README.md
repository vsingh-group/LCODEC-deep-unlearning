#  Deep Unlearning via Randomized Conditionally Independent Hessians (CVPR 2022)
#### Ronak Mehta, Sourav Pal, Sathya Ravi, Vikas Singh
![LCODEC Pipeline](/assets/lfoci_pipeline.png?raw=true)

## Abstract
Recent legislation has led to interest in machine unlearning, i.e., removing specific training samples from a predictive model as if they never existed in the training dataset. Unlearning may also be required due to corrupted/adversarial data or simply a user’s updated privacy requirement. For models which require no training (k-NN), simply deleting the closest original sample can be effective. But this idea is inapplicable to models which learn richer representations. Recent ideas leveraging optimization-based updates scale poorly with the model dimension d, due to inverting the Hessian of the loss function. We use a variant of a new conditional independence coefficient, L-CODEC, to identify a subset of the model parameters with the most semantic overlap on an individual sample level. Our approach completely avoids the need to invert a (possibly) huge matrix. By utilizing a Markov blanket selection common in the literature, we premise that L-CODEC is also suitable for deep unlearning, as well as other applications in vision. Compared to alternatives, L-CODEC makes approximate unlearning possible in settings that would otherwise be infeasible, including vision models used for face recognition, person re-identification and NLP models that may require unlearning data identified for exclusion.
[Full Paper Link at CVPR 2022 Proceedings.](https://openaccess.thecvf.com/content/CVPR2022/html/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.html)

## Code
All experiments are run within the specified folders, and call out to 'codec'.
Navigate to each folder for example scripts and directions on how to run in __expname__/README.md.

#### Conditional Independence Core
For our core conditional independence testing engine, you can check out and use the functions in the `codec/` folder.


#### Deep Learning Pipeline
For the deep learning unlearning pipeline, the `scrub/scrub_tools.py` file contains the main procedure. Our input perturbation revolves around the following at lines 145 and 188-192:
```
for m in range(params.n_perturbations):
	tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
	            acts, out = myActs.getActivations(tmpdata.to(device))
	            loss = criterion(out, y_true)
	            vec_acts = p2v(acts)
```
where the `getActivations` is computed using PyTorch activation hooks defined in `scrub/hypercolumn.py`.

## Reference
If you use this code, please cite our recent publication at CVPR 2022, which can be found here: [CVF CVPR 2022 Proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.html).


```
@InProceedings{Mehta_2022_CVPR,
    author    = {Mehta, Ronak and Pal, Sourav and Singh, Vikas and Ravi, Sathya N.},
    title     = {Deep Unlearning via Randomized Conditionally Independent Hessians},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10422-10431}
}
```
