import sys
import os
sys.path.append(os.path.abspath("ByteTrack"))

from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

def load_detections(detection_file):
    frame_dets = {}
    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            x1, y1, x2, y2, conf = map(float, parts[2:7])
            class_id = int(parts[7])
            det = [x1, y1, x2 - x1, y2 - y1, conf, class_id]
            frame_dets.setdefault(frame_id, []).append(det)
    return frame_dets

def run_tracker(detection_file, save_path, conf_thres=0.3):
    args = type('Args', (), {
        "track_thresh": conf_thres,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "aspect_ratio_thresh": 5.0,
        "min_box_area": 1,
        "mot20": False
    })()

    tracker = BYTETracker(args)
    detections = load_detections(detection_file)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as out_f:
        for frame_id in sorted(detections.keys()):
            frame_dets = np.array(detections[frame_id])
            if len(frame_dets) == 0 or np.isnan(frame_dets).any():
                continue

            online_targets = tracker.update(frame_dets, [1080, 1920], [1080, 1920])

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                out_f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},1,-1,-1,-1\n")

    print(f"✅ Takip tamamlandı: {save_path}")


if __name__ == "__main__":
    project_root = os.path.abspath(".")
    input_txt = os.path.join(project_root, "converted_outputs", "raw.txt")
    output_txt = os.path.join(project_root, "results", "raw", "mot.txt")

    print("\n🚀 Takip başlıyor: raw")
    run_tracker(input_txt, output_txt, conf_thres=0.3)
