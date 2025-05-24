import os
import glob

def yolo_to_bytetrack(input_label_dir, output_txt_path, image_width=1920, image_height=1080):
    label_files = sorted(glob.glob(os.path.join(input_label_dir, "*.txt")))
    print(f"\n🔍 {len(label_files)} adet .txt dosyası bulundu → {input_label_dir}")

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    total_lines = 0

    with open(output_txt_path, "w") as out_file:
        for label_file in label_files:
            filename = os.path.basename(label_file)
            name = os.path.splitext(filename)[0]

            try:
                # "M0101_img000001" → "000001" → frame_id = 1
                frame_str = name.split("img")[-1]
                frame_id = int(frame_str.lstrip("0") or "0")
            except Exception:
                print(f"⚠️ Frame ID çözümlenemedi: {filename}")
                continue

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        if len(parts) == 5:
                            class_id, x, y, w, h = map(float, parts)
                            conf = 1.0
                        else:
                            class_id, x, y, w, h, conf = map(float, parts[:6])
                    except ValueError:
                        continue

                    # normalize → pixel
                    x *= image_width
                    y *= image_height
                    w *= image_width
                    h *= image_height

                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2

                    out_file.write(
                        f"{frame_id},-1,{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{conf:.4f},{int(class_id)},1\n"
                    )
                    total_lines += 1

    print(f"✅ {output_txt_path} dosyasına {total_lines} satır yazıldı.")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "yolo_output", "raw", "labels")
    output_txt = os.path.join(project_root, "converted_outputs", "raw.txt")
    yolo_to_bytetrack(input_dir, output_txt)
