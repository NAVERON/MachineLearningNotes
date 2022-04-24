# 深度强化学习

## Q Learning算法

```
Initialize Q arbitrarily //随机初始化Q值
Repeat (for each episode): //每一次游戏，从小鸟出生到死亡是一个episode
    Initialize S //小鸟刚开始飞，S为初始位置的状态
    Repeat (for each step of episode):
        根据当前Q和位置S，使用一种策略，得到动作A //这个策略可以是ε-greedy等
        做了动作A，小鸟到达新的位置S'，并获得奖励R //奖励可以是1，50或者-1000
        Q(S,A) ← (1-α)*Q(S,A) + α*[R + γ*maxQ(S',a)] //在Q中更新S
        S ← S'
    until S is terminal //即到小鸟死亡为止
```

关于其中疑难点解释：

Ref : 
> [Do You Need Help Getting Started with Applied Machine Learning?](https://machinelearningmastery.com/start-here/)  
> [A Complete Guide on Getting Started with Deep Learning in Python](https://www.analyticsvidhya.com/blog/2016/08/deep-learning-path/)  







