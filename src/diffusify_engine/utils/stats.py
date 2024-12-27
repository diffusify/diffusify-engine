def format_stats_output(stats: dict) -> str:
    """Format processing statistics into a human-readable string.
    
    Args:
        stats: Dictionary containing processing statistics
        
    Returns:
        Formatted string containing statistics summary
    """
    def hr(char='-', length=50):
        return char * length

    # Helper function to format time if it exists
    def format_time(time_dict):
        if not time_dict:
            return "N/A"
        return time_dict.get('formatted', 'N/A')

    # Calculate success rate
    total = stats['processing_stats']['total_frames'] or 0
    processed = stats['processing_stats']['processed_frames']
    failed = len(stats['processing_stats']['failed_frames'])
    success_rate = (processed / total * 100) if total > 0 else 0

    output = [
        f"\n{hr('=', 60)}",
        "Video Processing Summary",
        f"{hr('=', 60)}",
        
        f"Input:  {stats['input_file']}",
        f"Output: {stats['output_file']}",
        
        f"\n{hr()}",
        "Video Information",
        f"{hr()}",
    ]

    # Add metadata if available
    if stats['metadata']:
        meta = stats['metadata']
        output.extend([
            f"Resolution: {meta.get('width', 'N/A')}x{meta.get('height', 'N/A')}",
            f"FPS: {meta.get('fps', 'N/A')}",
            f"Total Frames: {meta.get('total_frames', 'N/A')}",
            f"Audio: {'Yes' if meta.get('has_audio') else 'No'}"
        ])

    output.extend([
        f"\n{hr()}",
        "Processing Statistics",
        f"{hr()}",
        f"Frames Processed: {processed}/{total} ({success_rate:.1f}%)",
        f"Failed Frames: {failed}",
        f"GPU Memory Peak: {stats['processing_stats']['gpu_memory_peak']}",
        
        f"\n{hr()}",
        "Timing Information",
        f"{hr()}",
        f"Frame Extraction: {format_time(stats['timings']['extraction'])}",
        f"Frame Processing: {format_time(stats['timings']['processing'])}",
        f"Video Encoding: {format_time(stats['timings']['encoding'])}",
        f"{hr('=', 60)}\n"
    ])

    return "\n".join(output)