def compare_convention(path, this_dict):
    """指定pathのファイルの変更(New Warning, Fix Warning)の取得

    Args:
        path (path): 変更を取得するpath
        this_dict (dict): 取得した変更を{(修正数), (発生数)}の形で辞書型に変更

    Returns:
        dict: 辞書の出力
    """

    with open(path, "r") as f:
        lines = f.readlines()
        for target_line in lines:
            for id in this_dict.keys():
                if id in target_line:
                    this_dict[id][1] += 1
                if id in target_line and "Fix Warning" in target_line:
                    this_dict[id][0] += 1

    return this_dict
