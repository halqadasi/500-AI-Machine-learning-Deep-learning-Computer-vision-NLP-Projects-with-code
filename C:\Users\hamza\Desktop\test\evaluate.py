from detection_algorithms_gpu import ssim_outliers, ssim_outliers_rgb, mse_outliers, histogram_outliers, read_video_grayscale, read_video_rgb
from evaluation_metrics import load_ground_truth_from_csv, compare_algorithms
from pathlib import Path
from tqdm import tqdm
import csv
import torch

# Print device info at startup
print(f"\n{'='*80}")
print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"{'='*80}")

def test_best_algorithms_with_progress(videos_dir, csv_path, num_videos=None):
    """
    Test best performing algorithms on action category with progress bar.
    Automatically uses GPU if available, otherwise falls back to CPU.

    Args:
        videos_dir: directory containing videos
        csv_path: path to CSV file for one action
        num_videos: number of videos to test (None = all videos)
    """
    videos_dir = Path(videos_dir)

    # Get all video names
    video_names = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] not in video_names:
                video_names.append(row['video_id'])

    # Limit number of videos if specified
    if num_videos:
        video_names = video_names[:num_videos]

    action_name = Path(csv_path).stem
    print(f"\n{'='*80}")
    print(f"Testing {len(video_names)} videos from {action_name}")
    print(f"Algorithms: SSIM-Gray, SSIM-RGB, MSE, Histogram")
    print(f"{'='*80}\n")

    # Collect all data
    all_y_true = []
    all_predictions = {
        'SSIM-Gray (0.30)': [],
        'SSIM-Gray (0.40)': [],
        'SSIM-Gray (0.50)': [],
        'SSIM-RGB (0.30)': [],
        'SSIM-RGB (0.40)': [],
        'SSIM-RGB (0.50)': [],
        'MSE (500)': [],
        'MSE (1000)': [],
        'MSE (2000)': [],
        'Histogram (0.5)': [],
        'Histogram (0.7)': [],
    }

    # Process videos with progress bar
    for video_name in tqdm(video_names, desc="Processing videos", unit="video", ncols=100):
        video_path = videos_dir / video_name
        if not video_path.exists():
            continue

        # Load grayscale frames, RGB frames, and ground truth
        gray_frames = read_video_grayscale(video_path)
        rgb_frames = read_video_rgb(video_path)
        y_true = load_ground_truth_from_csv(csv_path, video_name)

        all_y_true.extend(y_true)

        # Run SSIM-Gray algorithms
        all_predictions['SSIM-Gray (0.30)'].extend(ssim_outliers(gray_frames, thr=0.30))
        all_predictions['SSIM-Gray (0.40)'].extend(ssim_outliers(gray_frames, thr=0.40))
        all_predictions['SSIM-Gray (0.50)'].extend(ssim_outliers(gray_frames, thr=0.50))

        # Run SSIM-RGB algorithms
        all_predictions['SSIM-RGB (0.30)'].extend(ssim_outliers_rgb(rgb_frames, thr=0.30))
        all_predictions['SSIM-RGB (0.40)'].extend(ssim_outliers_rgb(rgb_frames, thr=0.40))
        all_predictions['SSIM-RGB (0.50)'].extend(ssim_outliers_rgb(rgb_frames, thr=0.50))

        # Run MSE algorithms
        all_predictions['MSE (500)'].extend(mse_outliers(gray_frames, thr=500))
        all_predictions['MSE (1000)'].extend(mse_outliers(gray_frames, thr=1000))
        all_predictions['MSE (2000)'].extend(mse_outliers(gray_frames, thr=2000))

        # Run Histogram algorithms
        all_predictions['Histogram (0.5)'].extend(histogram_outliers(gray_frames, thr=0.5))
        all_predictions['Histogram (0.7)'].extend(histogram_outliers(gray_frames, thr=0.7))

    print(f"\n{'='*80}")
    print(f"Total: {len(all_y_true)} frames, {sum(all_y_true)} outliers")
    print(f"{'='*80}\n")

    # Compare algorithms
    results = compare_algorithms(all_y_true, all_predictions)

    # Print detailed confusion matrices for top 3
    print("\n" + "="*80)
    print("DETAILED CONFUSION MATRICES - TOP 3 ALGORITHMS")
    print("="*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results[:3], 1):
        print(f"\n{i}. {name}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   Confusion Matrix:")
        print(f"     TN: {metrics['true_negatives']:5d}  FP: {metrics['false_positives']:5d}")
        print(f"     FN: {metrics['false_negatives']:5d}  TP: {metrics['true_positives']:5d}")

    return results

if __name__ == "__main__":
    videos_directory = r"C:\Users\hamza\Desktop\test\UCF101_videos_modified"
    csv_file = r"C:\Users\hamza\Desktop\test\outlier_csvs\ApplyEyeMakeup.csv"

    # Test on all 25 videos in ApplyEyeMakeup action
    # Change num_videos to test different amounts:
    # - num_videos=10  -> first 10 videos
    # - num_videos=25  -> all 25 videos in ApplyEyeMakeup
    # - num_videos=None -> all videos in the CSV
    results = test_best_algorithms_with_progress(videos_directory, csv_file, num_videos=25)
