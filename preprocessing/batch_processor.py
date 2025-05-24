import os
import cv2

def apply_and_save_dataset(src_img_dir, dst_img_dir, filter_fns):
    os.makedirs(dst_img_dir, exist_ok=True)

    for filename in os.listdir(src_img_dir):
        if not filename.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(src_img_dir, filename)
        img = cv2.imread(img_path)

        for fn in filter_fns:
            img = fn(img)

        dst_path = os.path.join(dst_img_dir, filename)
        cv2.imwrite(dst_path, img)

    print(f"✅ Uygulandı → {dst_img_dir}")
