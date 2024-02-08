import os
def create_dir(create_path):
    """指定pathのディレクトリの作成

    Args:
        create_path (str): ディレクトリを作成したい場所へのpath
    """
    if not os.path.exists(create_path):
        os.makedirs(create_path)