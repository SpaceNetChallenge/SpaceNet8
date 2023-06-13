import os
import shutil
import tempfile

import fire


def main(src_path: str, dst_path: str, with_777: bool = True) -> None:
    print(src_path, "->", dst_path)

    if with_777:
        os.umask(0)

    if os.path.isdir(src_path):
        with tempfile.TemporaryDirectory(prefix="tmp_", dir=os.path.dirname(dst_path)) as name:
            shutil.copytree(src_path, name, dirs_exist_ok=True)
            if with_777:
                for root, _, files in os.walk(name):
                    os.chmod(root, 0o777)
                    for filename in files:
                        os.chmod(os.path.join(root, filename), 0o777)
            if os.path.exists(dst_path):
                # If a directory exists, shutil.move will move it into that directory, so once the existing
                # directory has been evacuated, run move, and then delete the old files
                with tempfile.TemporaryDirectory(prefix="tmp_", dir=os.path.dirname(dst_path)) as trash:
                    shutil.move(dst_path, trash)
                    shutil.move(name, dst_path)
            else:
                shutil.move(name, dst_path)
    else:
        with tempfile.NamedTemporaryFile(prefix="tmp_", dir=os.path.dirname(dst_path), mode="w", delete=False) as fp:
            shutil.copy(src_path, fp.name)
            if with_777:
                os.chmod(fp.name, 0o777)
            shutil.move(fp.name, dst_path)

    print("done")


if __name__ == "__main__":
    fire.Fire(main)
