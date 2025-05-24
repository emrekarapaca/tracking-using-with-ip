from batch_processor import apply_and_save_dataset
from image_processing import *


src_dir = "../data/raw/train/img"


apply_and_save_dataset(src_dir, "../data/blur/train/img", [apply_gaussian_blur])


apply_and_save_dataset(src_dir, "../data/hist_sharp/train/img", [apply_hist_eq, apply_sharpen])


apply_and_save_dataset(src_dir, "../data/bilateral_gamma/train/img", [apply_bilateral_filter, lambda img: apply_gamma_correction(img, 1.8)])
