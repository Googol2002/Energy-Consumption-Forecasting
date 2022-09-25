from dataset import LD2011_2014_summary_by_day
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]    # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def plot_ld2011_2014_summary_means_distribution():
    dataset = LD2011_2014_summary_by_day(length=4,
                                         csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv")

    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    fig.tight_layout(pad=5.0)

    expectations, variances = dataset.statistics()

    for i in range(4):
        y = np.asarray([sample[1][i * 24] for sample in dataset])
        axs[i].plot(range(len(y)), y)
        axs[i].title.set_text("第{}个分量".format(i * 24))
        axs[i].set_xlabel("天数")
        axs[i].set_ylabel("用电量")

    plt.show()
    print("期望:")
    print(expectations)
    print("方差")
    print(variances)


if __name__ == "__main__":
    plot_ld2011_2014_summary_means_distribution()
