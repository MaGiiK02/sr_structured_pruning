import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from lam.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from lam.SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from lam.SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from lam.SaliencyModel.attributes import attr_grad
from lam.SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from lam.SaliencyModel.BackProp import saliency_map_PG as saliency_map
from lam.SaliencyModel.BackProp import GaussianBlurPath
from lam.SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel
from utils.trainer import ForwardManager

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def lam(model, tensor_lr, window, scale = 4, sigma = 1.2, fold = 50, l = 9, alpha = 0.5):
    w, h, window_size = window
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=scale)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy, zoomin=scale)
    
    gini_index = gini(abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100
    print(f"The DI of this case is {diffusion_index}")
    
    return (diffusion_index, saliency_image_abs, saliency_image_kde, result)

def produce_lam_visualization(model, image_path, window, scale=4, half_input_size=False, alpha=0.5):
    w, h, window_size = window  # Define windoes_size of D
    if half_input_size and "urban" in image_path.lower(): w, h, = w//2, h//2
    img_lr, img_hr = prepare_images(image_path, scale, half_input_size and "urban" in image_path.lower())  # Change this image name
    tensor_lr = PIL2Tensor(img_lr)[:3]
    tensor_hr = PIL2Tensor(img_hr)[:3]
    
    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    patch_pos_pil = cv2_to_pil(draw_img)
    
    (diffusion_index, saliency_image_abs, saliency_image_kde, result) = lam(model, tensor_lr, (w, h, window_size),  alpha=alpha, scale=scale)

    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
 
    # Add Text to an image
    pil = make_pil_grid(
        [patch_pos_pil,
        saliency_image_abs,
        blend_abs_and_input,
        blend_kde_and_input,
        Tensor2PIL(torch.clamp(result, min=0., max=255.))]
    )
    
    return pil, diffusion_index

def multiple_patch_lam_visualization(model, image_path, patches, scale=4, half_input_size=False):
    visualizations = []
    diffusion_index = []
    for p in patches:
        pos_x, pos_y, window_size = p
        
        lam_vis, di = produce_lam_visualization(model, image_path, (pos_x, pos_y, window_size), scale, half_input_size)
        visualizations.append(lam_vis)
        diffusion_index.append(di)
        
    # Compose final image
    
    sizex, sizey = visualizations[0].size
    for img in visualizations:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    composed = Image.new('RGB', (sizex , sizey * len(visualizations)))
    top = 0
    for i in range(len(visualizations)):
        composed.paste(visualizations[i], (0, top))
        top += sizey
        
    return composed, visualizations, diffusion_index


