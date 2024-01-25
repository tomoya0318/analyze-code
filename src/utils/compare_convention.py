# 1つのファイルに対して，変更(New Warning, Delete Warning Code, Fix Warning, Neglect Warningの取得)
def compare_convention(path, this_dist):
    with open(path, "r") as f:
        lines = f.readlines()
        for target_line in lines:
            for id in this_dist.keys():
                if id in target_line:
                    this_dist[id][1] += 1
                if id in target_line and "Fix Warning" in target_line:
                    this_dist[id][0] += 1

    return this_dist
