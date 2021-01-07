def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - (elapsed_minutes * 60))
    return elapsed_minutes, elapsed_seconds
