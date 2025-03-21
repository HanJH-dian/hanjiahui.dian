# hanjiahui.dian
### ps：代码文件在master分支下

## 随机森林学习过程
### 阶段1：Python基础语法
学习内容：  
1. 变量与数据类型：整数、浮点数、字符串、布尔值。  
2. 基本运算：算术运算、逻辑运算（`and`/`or`/`not`）。  
3. 流程控制：  
    条件语句：`if`/`elif`/`else`。  
    循环语句：`for`循环、`while`循环。  
4. 函数与模块：  
    定义函数（`def`）、参数传递、返回值。  
    导入模块（如`import math`）。  

练习项目：  
 实现一个计算器（支持加减乘除和取模）。  
 编写猜数字游戏（随机生成数字，用户输入猜测）。



### 阶段2：Python数据结构与文件操作
目标：熟练操作列表、字典，理解文件读写。  
学习内容：  
1. 核心数据结构：  
    列表（`list`）：索引、切片、增删改查、列表推导式。  
    字典（`dict`）：键值对操作、遍历。  
    元组（`tuple`）与集合（`set`）。  
2. 文件操作：  
    读写文本文件（`open()`函数、`read()`/`write()`）。  

练习项目：  
 统计文本文件中单词的出现频率（输出前10高频词）。  
 实现一个学生成绩管理系统（用字典存储学生信息）。



### 阶段3：函数式编程与面向对象（OOP）
目标：掌握函数式编程思想，理解类与对象。  
学习内容：  
1. 函数进阶：  
    匿名函数（`lambda`）。  
    高阶函数（`map()`/`filter()`/`reduce()`）。  
2. 面向对象编程：  
    类（`class`）的定义与实例化。  
    属性（`self`）、方法（`def method()`）。  
    继承与多态（为后续实现决策树节点类打基础）。  

练习项目：  
 实现一个“银行账户”类（支持存款、取款、查询余额）。  
 用面向对象思想模拟简单的图书馆管理系统。



### 阶段4：NumPy与科学计算
目标：掌握NumPy数组操作，为随机森林实现打下基础。  
学习内容：  
1. NumPy基础：  
    创建数组（`np.array()`）、数组形状（`shape`）。  
    索引与切片（类似Python列表）。  
2. 数组运算：  
    广播机制（Broadcasting）。  
    常用函数：`np.sum()`、`np.mean()`、`np.random`模块。  
3. 矩阵操作：  
    矩阵乘法（`@`或`np.dot()`）、转置（`.T`）。  

练习项目：  
 用NumPy实现矩阵乘法（对比循环与向量化操作的效率差异）。  
 生成随机数据集并计算统计指标（均值、方差）。



### 阶段5：算法实现核心
目标：逐步实现决策树和随机森林，理解算法逻辑。  
学习内容：  
1. 决策树实现：  
    理解决策树的分裂逻辑（基尼不纯度计算）。  
    递归构建树结构（定义节点类）。  
2. 随机森林扩展：  
    Bootstrap采样（`np.random.choice`）。  
    特征子集随机选择（`max_features`参数）。  
3. 预测与评估：  
    多棵树结果聚合（投票或平均）。  
    OOB误差计算（用未采样的样本验证模型）。  

分步实践：  
1. 单棵决策树：  
    实现一个能分类Iris数据集的决策树。  
2. 随机森林：  
    集成多棵树，对比单棵树与森林的准确率差异。  



### 阶段6：特征重要性与可视化
目标：评估特征重要性并用Matplotlib绘图。  
学习内容：  
1. 特征重要性计算：  
    基于基尼不纯度减少量统计特征贡献。  
2. Matplotlib绘图：  
    绘制条形图（`plt.bar()`）、设置标题与标签。  

练习项目：  
 在随机森林中统计特征重要性，绘制排序后的条形图。  
 对比不同数据集（如波士顿房价）的特征重要性差异。



### 总结：
将随机森林拆解为决策树、Bootstrap、聚合预测等模块，逐个攻破。 


