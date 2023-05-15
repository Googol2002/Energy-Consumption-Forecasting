# The Oracle Project
## 最新进展

```
Val Error: 
 Accuracy: 96.989%, Avg loss: 0.028614
 Within the Power Generation: 96.875%
 Utilization Rate:  91.673%

Test Error: 
 Accuracy: 97.106%, Avg loss: 0.028301
 Within the Power Generation: 97.292%
 Utilization Rate:  91.308%
```
**Accuracy** 计算公式为: $\mathbb{E}_{x,y}\ {\left|\frac{\mathbf{model(x).mean} - y}{y}\right|}$.

**Within the Power Generation** 计算公式为: $\mathbb{E}_{x,y}\ {\mathbb{I}\left[y \leq \mathbf{model(x).mean} + c\times \sqrt{\mathbf{model(x).variance}}\right]}$，其中 $c\equiv1$.

**Utilization Rate** 计算公式为：$\mathbb{E}_{x,y}\ \left[{1-\frac{y}{\mathbf{model(x).mean} + c\times \sqrt{\mathbf{model(x).variance}}}}\right]$，其中 $c\equiv1$.


## 项目介绍
[2023-04]    顺利通过大学生创新创业训练计划中期答辩，并被评级为国家级项目 
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide1.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide2.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide3.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide4JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide5.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide6.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide7.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide8.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide9.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide10.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide11.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide12.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide13.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide14.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide15.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide16.JPG)
![Slides](https://github.com/Googol2002/Energy-Consumption-Forecasting/blob/main/figure/slides/Slide17.JPG)




<!-- ## 应用背景

随着极端气候与日俱增，世界对于“气候变暖”的重视程度已经达到了史无前例的高度。在气候变暖的主要诱因“碳排放”的诸多来源中，我国“能源电力”以40% 的“碳排放”占比一骑绝尘（第二名“建筑领域”占比约20% ）。事实上，大多数人对于火力发电厂的运行模式存在着一些误解，发电厂并非是按照当前实际用电量来确定发电量，而是通过预判未来的用电情况来计划发电量。仅以煤炭火电站为例，煤炭发电机组一次启动就需要一天之久，因此火电站必须预测第二天的用电情况。然而为保障居民用电与工业用电的安全稳定，**火电站不得不燃烧额外的煤炭，用于对抗预测中不可避免的误差。这些额外被燃烧的煤炭，不仅造成了资源的浪费，还导致了不必要的“温室气体”排放。**

## 项目预期
我们希望通过提高电力系统负荷预测的准确率，降低预测值与实际值之间波荡的方差，降低不必要的煤炭消耗，保护地球资源，减少温室气体排放。助力中国二氧化碳排放量于2030年前达到峰值，在2060年前实现碳中和。

## 技术实现

损失函数定义：
$$
\mathcal{L}(\theta)=\mathbb{E}_{x,y}\left[\frac{(\hat{y}-y)^2}{\hat{\sigma}^2}\right] + \alpha\times\sigma^2+\beta\times \|\theta\|_2\quad \\
\hat{y} = model(x).mean\quad \hat{\sigma}^2=model(x).variance
$$ -->


