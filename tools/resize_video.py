import ffmpeg
from pathlib import Path
import os
from tqdm import tqdm

def resize_videos_ffmpeg(input_folder, output_folder, target_width, target_height,
                         quality=18, preset='medium'):
    """
    Resize videos using ffmpeg while maintaining high-quality encoding.

    Args:
        input_folder: Path to the input video folder
        output_folder: Path to the output video folder
        target_width: Target width
        target_height: Target height
        quality: CRF quality value (0-51, lower is better quality, recommended 18-23)
        preset: Encoding speed preset (ultrafast, fast, medium, slow, veryslow)
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(input_folder).glob(f'*{ext}'))

    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    print(f"Found {len(video_files)} video files")
    print(f"Target size: {target_width}x{target_height}")
    print(f"Quality settings: CRF={quality}, Preset={preset}\n")

    # Process each video
    success_count = 0
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            output_path = os.path.join(output_folder, video_path.name)

            # Get video info (optional, for display)
            try:
                probe = ffmpeg.probe(str(video_path))
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                original_width = video_info['width']
                original_height = video_info['height']
            except:
                pass

            # Build ffmpeg processing pipeline
            stream = ffmpeg.input(str(video_path))

            # Video stream: resize
            video = stream.video.filter('scale', target_width, target_height)

            # Output settings
            stream = ffmpeg.output(
                video, output_path,
            )

            # Execute conversion (overwrite existing files, suppress verbose output)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            success_count += 1

        except ffmpeg.Error as e:
            print(f"\nError processing {video_path.name}:")
            print(f"  Details: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            print(f"\nError processing {video_path.name}: {str(e)}")

    print(f"\n{'='*50}")
    print(f"Done! Success: {success_count}/{len(video_files)}")
    print(f"Output directory: {output_folder}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resize videos using ffmpeg")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input video folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output video folder")
    parser.add_argument("--target_width", type=int, default=720, help="Target width")
    parser.add_argument("--target_height", type=int, default=480, help="Target height")
    parser.add_argument("--quality", type=int, default=18, help="CRF quality value (0-51, lower is better, recommended 18-23)")
    parser.add_argument("--preset", type=str, default="medium", help="Encoding speed preset (ultrafast, fast, medium, slow, veryslow)")
    args = parser.parse_args()

    resize_videos_ffmpeg(
        args.input_folder,
        args.output_folder,
        args.target_width,
        args.target_height,
        quality=args.quality,
        preset=args.preset,
    )