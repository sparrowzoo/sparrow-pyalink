import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt


def output_metrics(origin, pred):
    print("平均绝对误差 Mean absolute error=", round(sm.mean_absolute_error(origin, pred), 2))
    print("均方误差(最低） Mean squared error=", round(sm.mean_squared_error(origin, pred), 2))
    print("中位数绝对误差 Median absolute error=", round(sm.median_absolute_error(origin, pred), 2))
    print("解释方差分(最高 1.0) explained variance score=", round(sm.explained_variance_score(origin, pred), 2))
    print("R方得分（最高1.0，可能为负) R2 score", round(sm.r2_score(origin, pred), 2))


def plot_feature_importance(feature_importances, title, feature_names):
    # 将重要性能指标标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    # 让x 轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0])
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align="center")
    plt.xticks(pos, np.array(feature_names)[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.title(title, fontdict={"color": "white", "fontsize": 30})
    plt.tick_params(labelsize=10, labelcolor="white")
    plt.show()
