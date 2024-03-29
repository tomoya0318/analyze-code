import os


def get_file_name(directory):
    """ディレクトリ内の全てのファイルの名前の取得

    Args:
        directory (path): ファイル名を取得したいディレクトリへのpath

    Returns:
        list: ファイル名のリスト
    """
    FILE_NAME_LIST = []
    for filename in sorted(os.listdir(directory), key=lambda x: x.lower()):
        full_path = os.path.join(directory, filename)

        if os.path.isdir(full_path):
            FILE_NAME_LIST.append(filename)
            continue
        elif not filename.startswith("."):
            FILE_NAME_LIST.append(os.path.splitext(filename)[0])

    return FILE_NAME_LIST
