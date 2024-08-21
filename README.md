# Efficient super resolution deep-learning-based approach through pruning

Official Repository for my thesis on the development of a generic pruning framework for SR model.

## Abstract
Super Resolution refers to the task of reconstructing a bigger image from a smaller one,
producing a more crisp and detailed image. Such simple technique has found a wide va-
riety of applications, from medical imaging and satellite imaging reconstruction to image
enriching applications. The ability to produce higher-resolution images has far-reaching
implications and critical applications, which in some cases require real-time performances,
such as the case for video compression for streaming or real-time video super resolution. In
recent years, the use of Artificial Neural Networks for Super Resolution demonstrated much
higher quality in the reconstructed image but intractable computational times for real-time
applications, if compared to classical Super resolution approaches.
Neural Network pruning is a promising technique which has proven its capability in
removing parts of neural networks while marginally impacting the output quality. However,
the way in which pruning is executed does not translate in an effective improvement of the
model computational times. A subset of pruning techniques which does not present such
problem is known as Structured Pruning, which allows removing parts of a trained model
to obtain an actual computational speed-up in common hardware. While presenting such
interesting characteristics, Unstructured pruning techniques are much harder to implement
and often requires ad-hoc solutions for different models in which are implemented.
To solve such problems, we propose a generic structured pruning framework for Su-
per Resolution models, which can be applied to different Super Resolution Networks with
widely different architectures. We build such framework around a set of atomic pruning
schema implemented around a dependency graph, which enhances the understanding of how
to execute the atomic pruning schema on different super resolution models. We prove its
i
ability to reduce the model complexity by around 90% while retaining competitive PSNR
and SSIM scores.
