import sys
import traceback
import argparse
import torch

from .diffusify_engine.transformation_manager import TransformationManager
from .diffusify_engine.generative.generative_pipeline import GenerativePipeline

# example cmd arguments
# python -m src.main --mode transformation -i input.mp4 -o output.mp4
# python -m src.main --mode generative -p "a beautiful sunset"

def main():
    parser = argparse.ArgumentParser(description="Video processing and AI frame generation tool")
    parser.add_argument("--mode", 
        type=str, 
        required=True,
        choices=["transformation", "generative"],
        help="Select mode: 'transformation' for video processing, 'generative' for text-to-video")
    parser.add_argument("-i", "--input", 
        type=str, 
        required=False,
        help="Input video path (required for transformation mode)")
    parser.add_argument("-o", "--output", 
        type=str, 
        required=False,
        help="Output video path")
    parser.add_argument("-p", "--prompt",
        type=str,
        required=False,
        help="Text prompt for generative mode")
    args = parser.parse_args()

    # Add exit code tracking
    exit_code = 0

    try:
        if args.mode == "transformation":
            if not args.input:
                parser.error("--input is required for transformation mode")
            if not args.output:
                parser.error("--output is required for transformation mode")
            tm = TransformationManager(
                devices="cuda:0,cuda:1,cuda:2,cuda:3",
            )
            tm.process_video(
                input_path=args.input,
                output_path=args.output
            )
        elif args.mode == "generative":
            if not args.prompt:
                parser.error("--prompt is required for generative mode")
            gm = GenerativePipeline(
                args=args,
                devices="cuda:0,cuda:1,cuda:2,cuda:3",
            )
            # gm.generate(prompt...TODO)
    except Exception as e:
        print(f"Processing failed with error: {str(e)}")
        print("\nStack Trace:")
        traceback.print_exc()
        exit_code = 1
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
