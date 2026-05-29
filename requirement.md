# Requirement

我需要使用软件仿真来为我的机器学习模型提供合成数据。

我的研究对象是通过可测量的反作用力读取，估计可变形软体的内部性质。典型例子为，一块方形的可变形软体（"phantom"），内部嵌入了在一定位置，具有一定形状，和自身软硬的inclusions("lump")。用一个可读取接触反作用力的球形探头，沿法向低速（准静态）按压phantom，持续记录当前探头位置和力读取数值。在完成一定程度的扫描后，根据受力和空间关系，进行推理，得到推测的软体内部lump的信息。广义地来说，这是试图模仿人类触觉的工作原理。

按照你的建议1.用 Newton/VBD 建 phantom 2.用一个 kinematic sphere body 代替 Franka，沿法向慢速压入 3.直接想办法读取soft contact反力，这一步可以简化，最后内部运算总结输出只要z轴的力 4.扫描每个 (x, y) 点，记录 probe pose、indentation depth、总反力 Fz、可选接触 patch 特征。 5. 对每个样本随机生成 lump 位置、形状、刚度、大小，输出 label。 最终我还需要你搭建一个简单的validation U-net模型，读取过程数据，输出一个二维的0-1网格，表示是否检测到嵌入物。这个稍微修改的U-net实现可以参考：Reference/Palpation NN里的做法