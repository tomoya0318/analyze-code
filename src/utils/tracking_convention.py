import csv

#一つのcsvファイルからコーディング規約を取得
def tracking_convention(path):
    coding_convention_dist = {}
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            coding_convention_dist[row[0]] = [0, 0]
    coding_convention_dist = sorted(coding_convention_dist.items())
    coding_convention_dist = dict((x, y) for x, y in coding_convention_dist)
    return coding_convention_dist
