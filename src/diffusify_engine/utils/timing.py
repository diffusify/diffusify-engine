import time
from typing import Callable, Any, Dict

def format_duration(seconds: float) -> str:
    """Convert seconds into human readable time format.
    
    Args:
        seconds: Number of seconds to format
        
    Returns:
        str: Formatted duration string (e.g. "1h 30m 45.32s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f}s")
        
    return " ".join(parts)

def timed_execution(
    stats_dict: Dict[str, Any],
    func: Callable,
    stage_name: str
) -> Any:
    """Execute a function and time it.
    
    Args:
        func: Function to execute
        stage_name: Name of the processing stage
        stats_dict: Dictionary to store timing statistics
        
    Returns:
        Any: Result of the executed function
    """
    start_time = time.time()
    result = func()
    duration = time.time() - start_time
    
    # Store both raw and formatted duration
    stats_dict['timings'][stage_name] = {
        'raw_seconds': duration,
        'formatted': format_duration(duration)
    }
    
    # Print the formatted duration
    print(f"{stage_name.capitalize()} completed in {format_duration(duration)}")
    
    return result
