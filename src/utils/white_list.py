def lookup_white_list(txt_file):
    """ホワイトリストに設定されたプロジェクトの名称取得

    Args:
        txt_file (str): ホワイトリストを書いているファイルへのpath

    Returns:
        list: プロジェクトのリスト
    """

    # プロジェクトの選択
    project_list = []
    with open(txt_file, "r") as f:
        for line in f:
            project_list.append(line.strip())

    return sorted(project_list)
