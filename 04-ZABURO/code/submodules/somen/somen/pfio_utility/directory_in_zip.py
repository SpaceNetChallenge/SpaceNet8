# TODO: Update for pfio v2

# import os
# from typing import Optional

# import pfio


# def _get_abs_path(path: str) -> str:
#     if path.startswith("hdfs://"):
#         return path
#     # Resolve symlinks if Posix
#     return os.path.abspath(path)


# class DirectoryInZip:
#     def __init__(self, path: str) -> None:
#         parts = path.split(os.sep)
#         num_dot_zip = sum([name.endswith(".zip") for name in parts])

#         if num_dot_zip >= 2:
#             # TODO: message
#             raise ValueError

#         self.str = path
#         self.zip_container: Optional[pfio.containers.zip.ZipContainer]
#         self.root: str

#         if num_dot_zip == 0:
#             self.zip_container = None
#             self.root = _get_abs_path(path)
#         else:
#             split_point = path.find(".zip") + 4
#             self.zip_container = pfio.open_as_container(_get_abs_path(path[:split_point]))
#             self.root = path[split_point:].lstrip(os.sep)

#     def open(self, file_path: str, *args, **kwargs):
#         path = os.path.join(self.root, file_path)

#         if self.zip_container is not None:
#             return self.zip_container.open(path, *args, **kwargs)
#         else:
#             return pfio.open(path, *args, **kwargs)

#     def listdir(self, path: str = None):
#         path = self.root if path is None else os.path.join(self.root, path)

#         if self.zip_container is not None:
#             return self.zip_container.list(path)
#         else:
#             return pfio.list(path)

#     def __str__(self) -> str:
#         return self.str
