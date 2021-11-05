from rich.progress import Progress, BarColumn, TimeRemainingColumn

def build_training_progress_bar():
    return Progress(
        "[bold cyan][progress.description]{task.description}",
        BarColumn(bar_width=25),
        "[progress.percentage]{task.fields[batch_index]}/{task.total}",
        "- ETA: ",
        TimeRemainingColumn(),
        "- Generator Loss: {task.fields[generator_loss]:8.4f}",
        "- Discriminator Loss:{task.fields[discriminator_loss]:8.4f}"
    )