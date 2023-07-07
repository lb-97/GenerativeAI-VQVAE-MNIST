# VQVAE - MNIST

I took a deep dive into VQ-VAE code. Here's a little bit about VQ-VAE -

VQ-VAE is discretized VAE in latent space that helps in achieving high quality outputs. It varies from VAE by two points - use of discrete latent space, performing separate Prior training. VAE also showed impressive generative capabilities across data modalities - images, video, audio.

By using discrete latent space, VQ-VAE bypasses the 'posterior collapse' mode seen in traditional VAE. Posterior collapse is when latent space is not utilized properly and collapses to similar vectors independent of input, thereby resulting in not many variations when generating outputs.

Encoder, Decoder weights are trained along with L2 updates of embedding vectors. A categorical distribution is assumed of these latent embeddings and to truly capture the distribution of these vectors, these latents are further trained using PixelCNN model.

In the original paper, PixelCNN has shown to capture the distribution of data while also delivering rich detailing in generated output images. In the image space, PixelCNN decoder reconstructs a given input image with varying visual aspects such as colors, angles, lightning etc. This is achieved through autoregressive training with the help of masked convolutions. Auto regressive training coupled with categorical distribution sampling at the end of the pipeline facilitates PixelCNN to be an effective generative model.

A point to be noted here is that the prior of VQ-VAE is trained in latent space rather than image space through PixelCNN. So, it doesn't replace decoder as discussed in the original paper, rather trained independently to reconstruct the latent space. So, the first question that comes to my mind - How does latent reconstruction help in image generation? Is prior training required at all? What happens if not done?

I continued my experiments with VQ-VAE on MNIST data to see the efficacy of Prior training in the generated outputs. The output of encoder for every input image delivers a categorical index of a latent vector for every pixel in the output. As discussed previously, prior has been trained separately using PixelCNN (without any conditioning) in the latent space. If PixelCNN is a bunch of convolutions, then what makes it a generative model? This is an important question to ask and the answer to it is the sampling layer used on pixelCNN outputs during inference. The official code in Keras uses a tfp.layers.DistributionLambda(tfp.distributions.Categorical) layer as its sampling layer. Without this sampling layer PixelCNN outputs are deterministic and collapse to single output. Also similarly, sampling layer alone, i.e., without any PixelCNN trained prior, on the pre-determined outputs of encoder is deterministic. This is due to the fact that latent distances are correctly estimated by the pre-trained encoder and during inference categorical sampling layer would always sample the least distance latent, i.e., the one closest to the input. Therefore, the autoregressive nature of PixelCNN combined with a sampling layer for every pixel delivers an effective generative model. The outputs for all my experiments are shown in the image below -

![alt text](https://github.com/lb-97/dipy/blob/blog_branch/doc/posts/2023/assets/vq-vae-results.png)

Based on qualitatively analysis, PixelCNN outputs may require some extra work. This leads me to the next step in my research - to explore Diffusion models. 

References-
https://keras.io/examples/generative/vq_vae/
https://keras.io/examples/generative/pixelcnn/
