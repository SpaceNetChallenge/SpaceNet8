import multiprocessing


def setup_forkserver() -> None:
    # for https://github.com/pfnet/pfio/issues/123
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process()
    p.start()
    p.join()
