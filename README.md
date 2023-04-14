# TransGAN-noise

168行开始-----是在将数据传到generator之前，用VAE先将真实的数据集从encoder然后decoder , 然后把学习好的参数decoder更新到generator。
然后后面就是梯度添加噪音然后训练。
