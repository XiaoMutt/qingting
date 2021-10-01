import logging
import os


def setup_logger(log_folder: str, name: str):
    cwd = os.getcwd()

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    os.chdir(log_folder)
    res = logging.getLogger(name)

    res.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        f"[{name}: %(asctime)s %(filename)s:%(lineno)3d][%(levelname)-4.4s] %(message)s"
    )
    file_handler = logging.FileHandler(f"{name}.log", 'w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    res.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.DEBUG)
    res.addHandler(stream_handler)

    os.chdir(cwd)
    return res
