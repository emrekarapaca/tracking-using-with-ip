import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_tracking_file(file_path):
    track_data = {}
    total_boxes = 0

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            total_boxes += 1
            if track_id not in track_data:
                track_data[track_id] = []
            track_data[track_id].append(frame_id)

    total_ids = len(track_data)
    avg_track_len = round(sum(len(frames) for frames in track_data.values()) / total_ids, 2) if total_ids > 0 else 0
    return total_ids, total_boxes, avg_track_len

def main():
    input_dir = "results"
    variants = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    summary = []

    for variant in variants:
        mot_file = os.path.join(input_dir, variant, "mot.txt")
        if os.path.exists(mot_file):
            total_ids, total_boxes, avg_track_len = analyze_tracking_file(mot_file)
            summary.append({
                "variant": variant,
                "total_ids": total_ids,
                "total_boxes": total_boxes,
                "avg_track_len": avg_track_len
            })
            print(f"✅ {variant} completed")
        else:
            print(f"⚠️ File not found: {mot_file}")

    df = pd.DataFrame(summary)
    df.to_csv("track_analysis_summary.csv", index=False)
    print("\n📊 Tracking Analysis Summary:")
    print(df)


    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['variant'], df['avg_track_len'], color='skyblue')
    plt.title('Average Track Length per Variant')
    plt.xlabel('Image Variant')
    plt.ylabel('Average Track Length')
    plt.grid(axis='y')


    for bar, total_ids in zip(bars, df['total_ids']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                 f'IDs: {total_ids}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("track_analysis_chart.png")
    plt.show()

if __name__ == "__main__":
    main()
