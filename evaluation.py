import os
import cv2
import numpy as np
from openpyxl import Workbook
from skimage.metrics import structural_similarity as ssim

def compute_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def compute_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def compute_psnr(img1, img2, max_pixel_value=255.0):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

path = "./results/Pix2Pix_22"
all_files = os.listdir(path)
real_files = [f for f in all_files if "_real_B" in f]
fake_files = [f.replace("_real_B", "_fake_B") for f in real_files]

psnrs = []
ssims = []
mses = []
maes = []

wb = Workbook()
ws = wb.active
ws.append(['환자번호', '눈방향', 'SSIM', 'PSNR', 'MSE', 'MAE'])

for real_file, fake_file in zip(real_files, fake_files):
    real_img = cv2.imread(os.path.join(path, real_file))
    fake_img = cv2.imread(os.path.join(path, fake_file))

    assert real_img.shape == fake_img.shape, "Image shapes do not match!"

    psnr_value = compute_psnr(real_img, fake_img)
    ssim_value, _ = ssim(real_img, fake_img, full=True, multichannel=True, win_size=3)
    mse_value = compute_mse(real_img, fake_img)
    mae_value = compute_mae(real_img, fake_img)
    
    psnrs.append(psnr_value)
    ssims.append(ssim_value)
    mses.append(mse_value)
    maes.append(mae_value)

    # Extract patient and eye direction info from the filename
    parts = real_file.split('_')
    patient_id = parts[0]
    eye_direction = parts[2]
    
    ws.append([patient_id, eye_direction, ssim_value, psnr_value, mse_value, mae_value])
    #print(f"PSNR for {real_file} and {fake_file}: {psnr_value:.4f}")

avg_psnr = np.mean(psnrs)
avg_ssim = np.mean(ssims)
avg_mse = np.mean(mses)
avg_mae = np.mean(maes)

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average MSE: {avg_mse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")

wb.save('result_Pix2Pix_22.xlsx')
