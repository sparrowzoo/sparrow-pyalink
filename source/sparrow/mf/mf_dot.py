import numpy

if __name__ == '__main__':  # 主函数
    R = [
        [1, 1],
        [4, 2],
        [1, 3]
    ]
    R = numpy.array(R)

    R_MF = numpy.dot(R, R.T)

    print(R_MF)  # 输出新矩阵

