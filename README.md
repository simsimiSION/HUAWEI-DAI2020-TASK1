## DAI2020 SMARTS Competition Track 1:  Single Agent, Multi-Lane Cruising

### 文件说明

* starter_kit: 包含训练程序、测试程序
    * submit_example: 提交样本
    * train_example：训练样本
    * **.zip：以提交样本，名称为实际结果
    * other：附属文件
* dataset_public: 官方训练场景
* dataset_demo：自定义训练场景

### 程序执行

#### 环境开启

```shell
# 启动docker
docker run -itd -p 6006:6006 -p 8081:8081 -p 8082:8082 -v /Users/qaq/Desktop/HUAWEI-DAI-TASK1:/home --name smarts smartt/dai:task1 bash
docker exec -it smarts bash

# 编译场景(只需执行一次)
scl scenario build-all dataset_publce

# 开启场景可视化
scl envision start -s dataset_public
```

#### 训练

```shell l
cd train_example

# 训练(horizon设置为1500，1000有些场景跑不完)
python train_multi_scenario.py --headless --horizon 1500 --num_workers 7
```

#### 测试

```shell
cd submit_example

# 使用docker cp将保存的checkpoint复制到该文件夹下

# 测试
python run.py
```

