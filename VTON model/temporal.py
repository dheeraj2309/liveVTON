# --- START OF FILE inference_with_warp_gen.py ---

import shutil
from pathlib import Path
import mediapipe as mp
import cupy
import torch
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw

# <<< NEW: Import for CUDA-accelerated OpenCV functions >>>
import cv2.cuda 

import torchvision.transforms as transforms
from pipelines.dmvton_pipeline_warp import DMVTONPipeline
from utils.torch_utils import select_device,get_ckpt, load_ckpt
from models.generators.mobile_unet import MobileNetV2_unet
# from models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from models.generators.res_unet import ResUnetGenerator
from models.warp_modules.afwm import AFWM
from Real_ESRGAN.inference_realesrgan_copy import run_realesrgan

def make_power_2(img, base=16, method=Image.BICUBIC):
    """Resize image so that both dimensions are multiples of base."""
    try:
        ow, oh = img.size  # PIL image
    except Exception:
        oh, ow = img.shape  # numpy image
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def transform_image(img, method=Image.BICUBIC, normalize=True):
    """
    Apply transformation pipeline similar to get_transform().
    Steps:
        - Resize to power of 2
        - Random horizontal flip (only in train mode)
        - Convert to tensor
        - Normalize to [-1, 1] if normalize=True
    """
    base = float(2**4)
    img = make_power_2(img, base, method)
    img = transforms.ToTensor()(img)
    if normalize:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img
def resize_with_aspect(image, target_size=(256, 192), pad_color=(255, 255, 255)):
    """
    Resize image to target_size while maintaining aspect ratio and padding.
    target_size: (height, width)
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]

    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with the scale
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded
def crop_person(image, mp_selfie_segmentation):
    h, w, _ = image.shape
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as segmentor:
        results = segmentor.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask > 0.5  # Boolean mask (True=person)

        # Create a white background
        white_bg = np.ones_like(image, dtype=np.uint8) * 255  # White image

        # Apply the mask: keep person where mask=True, else white
        segmented_person = np.where(mask[:, :, None], image, white_bg)
        
        # Find bounding box of the person
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return image  # No person found
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        box_w = x_max - x_min
        box_h = y_max - y_min

        # Only adjust if smaller than 192x256
        target_w, target_h = 192, 256
        if box_w < target_w or box_h < target_h:
            # Adjust width
            if box_w < target_w:
                pad_w = (target_w - box_w) // 2
                x_min = max(0, x_min - pad_w)
                x_max = min(w, x_min + target_w)

            # Adjust height
            if box_h < target_h:
                pad_h = (target_h - box_h) // 2
                y_min = max(0, y_min - pad_h)
                y_max = min(h, y_min + target_h)

            # Ensure final crop fits within image bounds
            x_min = max(0, min(x_min, w - target_w))
            x_max = min(w, x_min + target_w)
            y_min = max(0, min(y_min, h - target_h))
            y_max = min(h, y_min + target_h)

        cropped = segmented_person[y_min:y_max, x_min:x_max]
        return cropped



def process_photo(img_dir,cloth_name,device,pipeline=DMVTONPipeline()):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    warp_model = AFWM(3, True).to(device)
    gen_model = ResUnetGenerator(7, 4,5).to(device)
    
    warp_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_warp_last.pt")
    load_ckpt(warp_model, warp_ckpt)
    warp_model.eval()

    gen_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_gen_last.pt")
    load_ckpt(gen_model, gen_ckpt)
    gen_model.eval()

    upscaler=run_realesrgan(gpu_id=0)
    print("models loaded")

    with torch.no_grad():

        person_img=cv2.imread(img_dir)
        person_img=crop_person(person_img,mp_selfie_segmentation)
        seg_img=person_img
        person_img=resize_with_aspect(person_img,target_size=(256, 192))
        person_img=cv2.cvtColor(person_img,cv2.COLOR_BGR2RGB)
        person_img=Image.fromarray(person_img).convert('RGB')
        person_img=transform_image(person_img).unsqueeze(0).to(device)

        cloth_img=cv2.imread(f"cloth/{cloth_name}.jpg")
        cloth_img=resize_with_aspect(cloth_img,target_size=(256, 192))
        cloth_img=cv2.cvtColor(cloth_img,cv2.COLOR_BGR2RGB)
        cloth_img=Image.fromarray(cloth_img).convert('RGB')
        cloth_img=transform_image(cloth_img).unsqueeze(0).to(device)

        cloth_mask=cv2.imread(f"cloth_mask/{cloth_name}.jpg")
        cloth_mask=resize_with_aspect(cloth_mask,target_size=(256, 192))
        cloth_mask=cv2.cvtColor(cloth_mask,cv2.COLOR_BGR2RGB)
        cloth_mask=Image.fromarray(cloth_mask).convert('L')
        cloth_mask=transform_image(cloth_mask,method=Image.NEAREST, normalize=False).unsqueeze(0).to(device)
        print("data transformed")



        with cupy.cuda.Device(int(device.split(':')[-1])):
 
            p_tryon, warped_cloth,warped_mask = pipeline(warp_model,gen_model,person_img, cloth_img, cloth_mask, phase="test")
        img_tensor = p_tryon[0].squeeze()
        img_tensor1 = warped_cloth[0].squeeze()
        # Process the warped mask tensor
        img_tensor2 = warped_mask[0].squeeze()

        # Convert to numpy and process
        cv_img = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
        cv_img = (cv_img * 255).astype(np.uint8)
        
        cv_img1 = (img_tensor1.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
        cv_img1 = (cv_img1 * 255).astype(np.uint8)
        
        # Convert the single-channel mask tensor to a saveable image
        cv_img2 = (img_tensor2.detach().cpu().numpy() * 255).astype(np.uint8)

        p_tryon = cv_img
        warped_cloth = cv_img1
        warped_mask_img = cv_img2
        
        print("tryon generated")
        
        _, _, output = upscaler.enhance(p_tryon, has_aligned=False, only_center_face=False, paste_back=True)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        _, _, output1 = upscaler.enhance(warped_cloth, has_aligned=False, only_center_face=False, paste_back=True)
        output1 = cv2.cvtColor(output1, cv2.COLOR_RGB2BGR)
        print("upscaled")
        
        img_save_name = os.path.splitext(os.path.basename(img_dir))[0]

        # Save all three results: try-on, warped cloth, and warped mask
        cv2.imwrite(f"results/{img_save_name}_out.jpg", output)
        cv2.imwrite(f"results/{img_save_name}_out_warp.jpg", output1)
        cv2.imwrite(f"results/{img_save_name}_out_mask.jpg", warped_mask_img)
        cv2.imwrite(f"results/{img_save_name}_seg.jpg", seg_img)
        
        print("saved")
def process_vid(vid_dir,cloth_name,device,pipeline=DMVTONPipeline()):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    warp_model = AFWM(3, True).to(device)
    gen_model = ResUnetGenerator(7, 4,5).to(device)
    
    warp_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_warp_last.pt")
    load_ckpt(warp_model, warp_ckpt)
    warp_model.eval()

    gen_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_gen_last.pt")
    load_ckpt(gen_model, gen_ckpt)
    gen_model.eval()

    upscaler=run_realesrgan(gpu_id=0)
    print("models loaded")

    with torch.no_grad():



        cloth_img=cv2.imread(f"cloth/{cloth_name}.jpg")
        cloth_img=resize_with_aspect(cloth_img,target_size=(256, 192))
        cloth_img=cv2.cvtColor(cloth_img,cv2.COLOR_BGR2RGB)
        cloth_img=Image.fromarray(cloth_img).convert('RGB')
        cloth_img=transform_image(cloth_img).unsqueeze(0).to(device)

        cloth_mask=cv2.imread(f"cloth_mask/{cloth_name}.jpg")
        cloth_mask=resize_with_aspect(cloth_mask,target_size=(256, 192))
        cloth_mask=cv2.cvtColor(cloth_mask,cv2.COLOR_BGR2RGB)
        cloth_mask=Image.fromarray(cloth_mask).convert('L')
        cloth_mask=transform_image(cloth_mask,method=Image.NEAREST, normalize=False).unsqueeze(0).to(device)
        print("data transformed")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"Error: Cannot open video {vid_dir}")
            return

    # Video writer setup
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_save= os.path.splitext(os.path.basename(vid_dir))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"results/{vid_save[0]}_out{vid_save[1]}", fourcc, fps, (768,1024))

        while True:
            ret, orig_frame = cap.read()
            if not ret:
                break


            person_img=crop_person(orig_frame,mp_selfie_segmentation)
            person_img=resize_with_aspect(person_img,target_size=(256, 192))
            person_img=cv2.cvtColor(person_img,cv2.COLOR_BGR2RGB)
            person_img=Image.fromarray(person_img).convert('RGB')
            person_img=transform_image(person_img).unsqueeze(0).to(device)

            with cupy.cuda.Device(int(device.split(':')[-1])):
    
                p_tryon, warped_cloth,warped_mask = pipeline(warp_model,gen_model,person_img, cloth_img, cloth_mask, phase="test")
            img_tensor = p_tryon[0].squeeze() # Take the first image in the batch

    # Convert to numpy and process
            cv_img = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2  # Normalize to [0, 1]
            cv_img = (cv_img * 255).astype(np.uint8)  # Scale to [0, 255]



            # Convert RGB to BGR for OpenCV
            # p_tryon = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            # p_tryon=cv_img

            print("tryon generated")
            # _, _, output = upscaler.enhance(p_tryon, has_aligned=False, only_center_face=False, paste_back=True)
            output,_ = upscaler.enhance(cv_img, outscale=2)
            # output=p_tryon
            output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
            out.write(output)



def real_time(cloth_name,device,pipeline=DMVTONPipeline()):
    # --- Model Loading ---
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    warp_model = AFWM(3, True).to(device)
    gen_model = ResUnetGenerator(7, 4,5).to(device)
    
    warp_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_warp_last.pt")
    load_ckpt(warp_model, warp_ckpt)
    warp_model.eval()

    gen_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_gen_last.pt")
    load_ckpt(gen_model, gen_ckpt)
    gen_model.eval()
    
    # <<< MODIFIED: Removed upscaler from real-time to prioritize speed >>>
    # upscaler=run_realesrgan(gpu_id=0) 
    print("models loaded")

    with torch.no_grad():
        # --- Cloth Pre-processing ---
        cloth_img=cv2.imread(f"cloth/{cloth_name}.jpg")
        cloth_img=resize_with_aspect(cloth_img,target_size=(256, 192))
        cloth_img=cv2.cvtColor(cloth_img,cv2.COLOR_BGR2RGB)
        cloth_img=Image.fromarray(cloth_img).convert('RGB')
        cloth_img=transform_image(cloth_img).unsqueeze(0).to(device)

        cloth_mask=cv2.imread(f"cloth_mask/{cloth_name}.jpg")
        cloth_mask=resize_with_aspect(cloth_mask,target_size=(256, 192))
        cloth_mask=cv2.cvtColor(cloth_mask,cv2.COLOR_BGR2RGB)
        cloth_mask=Image.fromarray(cloth_mask).convert('L')
        cloth_mask=transform_image(cloth_mask,method=Image.NEAREST, normalize=False).unsqueeze(0).to(device)
        print("data transformed")

        # --- Real-time Loop Setup ---
        cap = cv2.VideoCapture(0)
        
        # <<< NEW: State variables for optical flow >>>
        prev_gray_gpu = None
        prev_tryon_output = None
        
        # <<< NEW: Optical Flow Calculator Setup (GPU-accelerated) >>>
        flow_calculator = cv2.cuda.FarnebackOpticalFlow_create(5, 0.5, False, 15, 3, 5, 1.2, 0)
        
        # <<< NEW: Blending factor for smoothing >>>
        # A higher alpha relies more on the newly generated frame (less smooth, more responsive).
        # A lower alpha relies more on the warped previous frame (smoother, but can lag).
        alpha = 0.7 

        while True:
            ret, orig_frame = cap.read()
            if not ret:
                break
            
            # --- Person Image Pre-processing ---
            person_img_cropped = crop_person(orig_frame, mp_selfie_segmentation)
            person_img_resized = resize_with_aspect(person_img_cropped, target_size=(256, 192))
            
            person_img_rgb = cv2.cvtColor(person_img_resized, cv2.COLOR_BGR2RGB)
            person_img_pil = Image.fromarray(person_img_rgb).convert('RGB')
            person_img_tensor = transform_image(person_img_pil).unsqueeze(0).to(device)

            # --- VTON Generation ---
            with cupy.cuda.Device(int(device.split(':')[-1])):
                p_tryon, _, _ = pipeline(warp_model, gen_model, person_img_tensor, cloth_img, cloth_mask, phase="test")
            
            img_tensor = p_tryon[0].squeeze()
            cv_img_rgb = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
            current_tryon_output_rgb = (cv_img_rgb * 255).astype(np.uint8)
            current_tryon_output = cv2.cvtColor(current_tryon_output_rgb, cv2.COLOR_RGB2BGR) # Now in BGR for OpenCV
            
            print("tryon generated")
            
            # <<< NEW: Temporal Smoothing Logic using Optical Flow >>>
            # 1. Prepare current frame for flow calculation (use the resized BGR frame)
            gray = cv2.cvtColor(person_img_resized, cv2.COLOR_BGR2GRAY)
            current_gray_gpu = cv2.cuda_GpuMat()
            current_gray_gpu.upload(gray)

            # Default output is the newly generated frame
            final_output = current_tryon_output

            # 2. If we have a previous frame, calculate flow and blend
            if prev_gray_gpu is not None and prev_tryon_output is not None:
                # Calculate flow on GPU
                flow_gpu = flow_calculator.calc(prev_gray_gpu, current_gray_gpu, None)
                
                # Download flow to CPU for cv2.remap
                flow_cpu = flow_gpu.download()

                # Warp the previous try-on result using the calculated flow
                warped_prev_output = cv2.remap(prev_tryon_output, flow_cpu, None, cv2.INTER_LINEAR)
                
                # Blend the current result with the warped previous result
                final_output = cv2.addWeighted(current_tryon_output, alpha, warped_prev_output, 1 - alpha, 0)

            # --- Display and Update State ---
            display_output = cv2.resize(final_output, (192*2, 256*2))
            
            # <<< MODIFIED: Show both original and smoothed VTON for comparison >>>
            cv2.imshow("Smoothed VTON", display_output)
            cv2.imshow("Original Feed", orig_frame)

            if cv2.waitKey(1) & 0xFF == ord("d"):
                break
            
            # <<< NEW: Update state for the next frame >>>
            # The current grayscale frame becomes the previous one for the next iteration.
            prev_gray_gpu = current_gray_gpu 
            # The final blended output becomes the previous output for the next iteration.
            prev_tryon_output = final_output

        cap.release()
        cv2.destroyAllWindows()

            
# e.g., 'image'




def main():
    # Device
    device = select_device(0)

    # Inference Pipeline
    # process_photo("test13.jpg","00515_00",device)
    # process_vid('8166002-hd_720_1366_25fps.mp4',"00019_00",device)
    real_time("00515_00",device)
    # process_type=input()
    # if process_type=="image":
    #     process_photo("a.jpg","00033_00",pipeline,device)
    # elif process_type=="video":
    #     process_vid()



    # Dataloader
    # test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    # data_loader = DataLoader(
    #     test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers
    # )

    # run_test_pf(
    #     pipeline=pipeline,
    #     data_loader=data_loader,
    #     device=device,
    #     log_path=log_path,
    #     save_dir=opt.save_dir,
    #     img_dir=Path(opt.dataroot) / 'test_img',
    #     save_img=True,
    # )


if __name__ == "__main__":
    main()

    # opt = TestOptions().parse_opt()
    # main(opt)