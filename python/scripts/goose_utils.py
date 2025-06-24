import os, shutil, cv2 

def refactor_data(source_root: str,target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    found = False
    for root, _, files in os.walk(source_root):
        for fname in files:
            if "windshield_vis" in fname.lower():
                found= True
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(target_dir, fname)
                shutil.copy2(src_path, dst_path)

    if found: 
        print("Files found!")

    return True

def images2video(img_folder, output_path, fps=30, resize:tuple=None):

    images = sorted([
        f for f in os.listdir(img_folder)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ])

    if not images:
        print(f"No images found in folder {img_folder}!!")
        return 
    
    first_image_path = os.path.join(img_folder, images[0])
    first_image = cv2.imread(first_image_path)

    if resize:
        width, height = resize[1], resize[0]  # (W, H)
    else:
        height, width = first_image.shape[:2]
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, frameSize=(width, height))

    for img_name in images: 
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping invalid image : {img_name}")
            continue
        if resize:
            img = cv2.resize(img, (width, height))
        else:
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
        
        video.write(img)

    video.release()
    print(f"Video saved to:{output_path}")


if __name__ == '__main__': 

    # =======  seperate RGB from NIR  ========== #
    # source_root = "data/GOOSE/test/images/test/"
    # target_dir = "data/GOOSE/test/input/" 

    # if refactor_data(source_root, target_dir):
    #     print(f"Test images saved to: {target_dir}")

    # ======= Get a video from test images =======# 
    video_name = "out_masks.mp4"
    images_folder = "data/GOOSE/test/output/"
    output_path = "data/GOOSE/test/video/"+video_name
    images2video(images_folder, output_path, fps=5, resize=(600,600))
        

        