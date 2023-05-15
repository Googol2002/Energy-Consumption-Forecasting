import json

from dataset.ETT_data import Dataset_ETT_hour

if __name__ == "__main__":
    datasets = [Dataset_ETT_hour(root_path='dataset/ETT-small', timeenc=0, scale=True,
                                 inverse=False,  features='S', target='OT', freq='h',
                                 flag=f, data_path='ETTh2.csv',
                                 size=[24 * 4 * 4, 0, 24 * 4], window=24)
                for f in ['train', 'val', 'test']]
    datas = list()
    for i in range(0, 1000, 15):
        x, y, _, _ = datasets[0][i]
        datas.append({"X": list(x.reshape(-1)), "Y": list(y.reshape(-1))})

    with open(r"data.json", "w") as out:
        json.dump(datas, out)