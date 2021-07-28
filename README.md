# cyclegan-learning
My first attempt to implement a CycleGAN in TensorFlow based on Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", in IEEE International Conference on Computer Vision (ICCV), 2017.  See: https://junyanz.github.io/CycleGAN/ 

## Current status
The model seems to work, but iterations take quite a long time so it's hard to verify.  I do not think the slowness is related to a bug, but it's worth another look. Part of the issue is the very large increase in parameters when the residual block concatenates its input rather than adds to it.  The paper does not make it clear which approach they chose (addition vs concatenation).  Either way, I think I will need to devote quite a bit of computation time to check how well it works.

![My image](https://github.com/ultraglorious/cyclegan-learning/blob/main/output/horse2zebra/image_at_epoch_0014.png)
