
# 总体调用部件
import time
from tkinter import Tk, Canvas
import numpy as np
from Ship import Ship


class Viewer():
    
    ships_count = 10
    state_dim = 2 + 4*5   # 自身属性(航速和目标距离) + 4个领域 * 每个领域属性
    action_dim = 2

    course_bound = [-5, 5]    #目标方向偏差
    speed_bound = [-1, 1]
    # num_iterations = 10000
    dis = 250
    
    def __init__(self):
        self.tk = Tk()
        
        self.window_width = 800
        self.window_height = 600
        
        self.canvas = Canvas(self.tk, width=self.window_width, height=self.window_height)
        
        self.ships = {}
        self.all_observations = {}  # 以自定形式存储数据   id : observation
#         for _ in range(10):
#             self.ships.append(self.createRandomEntity())
        self.canvas.pack()
        self.tk.title("Simulation of USVs")
        self.render()
    
    def createRandomEntity(self):
        position_list = np.array(np.random.random((1, 2))).flatten()
        position = np.multiply(position_list, self.window_height)
        
        velocity_list = np.array(-1 + 2*np.random.random((1, 2))).flatten()
        velocity = np.multiply(velocity_list, 5)
        entity = Ship(position, velocity, self.window_width, self.window_height)
        #print(entity.toString())  ###############################################################################3
        time.sleep(0.01)
        return entity
    #  获取周边放到了Ship中，方便逻辑调用
    def render(self):  # 根据当前状况绘制
        self.canvas.delete("all")
        
        for k, v in self.ships.items():
            s = v
            if k == self.train_id:
                self.canvas.create_oval(s.position[0]-10, self.window_height-s.position[1]-10, s.position[0]+10, self.window_height-s.position[1]+10, fill="red")
                self.canvas.create_text(s.destination[0], self.window_height-s.destination[1], text = str("O"), fill = "red")
            else:
                self.canvas.create_oval(s.position[0]-10, self.window_height-s.position[1]-10, s.position[0]+10, self.window_height-s.position[1]+10, fill="blue")
                self.canvas.create_text(s.destination[0], self.window_height-s.destination[1], text = str("O"), fill = "green")
            # 方向
            self.canvas.create_line(s.position[0], self.window_height-s.position[1], s.position[0]+s.velocity[0]*10, self.window_height-s.position[1]-s.velocity[1]*10, fill="black")
            
            # 绘制历史轨迹
            for i in range(len(s.history)):
                his = s.history[i]
                self.canvas.create_text(his[0], self.window_height-his[1], text="*")
            
        self.tk.update()
    
    last_dis_destination = 1000
    cur_dis_destination = 1000
    def step(self,  **actions):    # 这里传入每一个对象的动作，每一艘船舶都向前走一步，之后会得到新的环境
        # 这里先做动作，舵角，速度变化等
        # print("在 环境中step打印当前传入的动作   ", actions)
        train_reward = 0
        done = False
        #action = None
        # 根据action做出动作
        for key, action in actions.items():           #  重点：一个是环境获取，一个是惩罚奖励设置函数
            s = self.ships[key]
            if s.isDead:       #  如果已经死亡，则不进行动作指导
                continue
            #print("动作", action)
            action = np.array([np.clip(action[0], self.course_bound[0], self.course_bound[1]), np.clip(action[1], self.speed_bound[0], self.speed_bound[1])])
#             if key == self.train_id:
#                 print("action id:", key, ", action:", action)   # -2 2/////-0.2  0.2
            # 根据id操作相应的动作，修改数据
            s.courseTurn(action[0])  #动作1是改变舵角   动作2 是改变速度
            s.speedChange(action[1])
            
        # 判断是否碰撞
        for key, ship in self.ships.items():
            if ship.isDead:
                continue
            
            ship.goAhead()
            ship.getNear(self.dis, **self.ships)
            ship.isCollision()
        # 根据碰撞情况制定惩罚奖励  reward  ################# 规则遵守情况奖励设计
        train_ship = self.ships[self.train_id]
        self.cur_dis_destination = train_ship.dis_Destination()
        train_action = actions[self.train_id]
        train_observation = self.all_observations[self.train_id]
        #print("提取当前的环境", train_observation)   # 2  7  12  17
        if train_ship.isDead:
            # 如何造成的碰撞，追究原因，给予惩罚
            done = True
            
            speed = train_ship.getSpeed()
            if  speed > 6:
                train_reward -= 0.1
            
#             if train_observation[2] > 0 and train_action[0] > 0:
#                 train_reward -= 0.1
#             elif train_observation[7] > 0 and train_action[0] > 0:
#                 train_reward -= train_action[0]
#             elif train_observation[17] > 0 and train_action[0] < 0:
#                 train_reward += train_action[0]
#             else:
#                 train_reward -= 0.5
            train_reward -= 1
        else:
            # 会遇态势，如果遵守规则，奖励多一些，否则给予奖励少一些
            done = False
            
            des_dis = self.last_dis_destination - self.cur_dis_destination
            if not train_ship.now_near and des_dis < 100:
                train_reward += des_dis/10
            
#             if train_observation[2] > 0 and train_action[0] > 0:
#                 train_reward += 0.2
#             elif train_observation[7] > 0 and train_action[0] < 0:
#                 train_reward -= train_action[0]
#             elif train_observation[17] > 0 and train_action[0] > 0:
#                 train_reward += train_action[0]
#             else:
#                 train_reward += 0.5
            train_reward += 0.5
            
        self.last_dis_destination = self.cur_dis_destination
        self.all_observations.clear()
        for k, v in self.ships.items():
            self.all_observations[k] = v.getObservation(self.dis, **self.ships)
        self.render()  #渲染当前画面 =====可以在外层调用，也可以直接放在步进合并渲染
        time.sleep(0.001)
        
        return self.all_observations, train_reward, done   # 观察值， 奖励， 一个回合是否完成
        pass
    
    def setReward(self):
        
        pass
    
    def saveAllShips(self):
        for id, ship in self.ships.items():
            ship.storeTrajectories()
        pass
    def reset(self):  # 重置环境和变量的条件
        print("一个回合结束，重新生成新的环境")
        self.train_id = None
        
        self.last_dis_destination = 1000
        self.cur_dis_destination = 1000
        self.ships.clear()
        self.all_observations.clear()
        self.canvas.delete("all")
        # 重新生成一个新的环境
#         for _ in range(self.ships_count):
#             temp = self.createRandomEntity()
#             self.ships[temp.id] = temp
#       
        # 对遇态势
        temp = Ship(np.array([500.0, 100.0]), np.array([0.0, 2.5]), width=self.window_width, height=self.window_height)
        self.ships[temp.id] = temp
        time.sleep(0.01)
         
        temp = Ship(np.array([500.0, 500.0]), np.array([0.0, -2.2]), width=self.window_width, height=self.window_height)
        self.ships[temp.id] = temp
        time.sleep(0.01)
        #对遇和 左舷交叉相遇     3 无人艇会遇
        temp = Ship(np.array([200.0, 300.0]), np.array([1.2, -1.3]), width=self.window_width, height=self.window_height)
        self.ships[temp.id] = temp
        time.sleep(0.01)
        # 四无人艇   会遇
        temp = Ship(np.array([700.0, 450.0]), np.array([-2.0, -3]), width=self.window_width, height=self.window_height)
        self.ships[temp.id] = temp
        time.sleep(0.01)
        #再来一个追越
        temp = Ship(np.array([600.0, 15.0]), np.array([-2, 1.0]), width=self.window_width, height=self.window_height)
        self.ships[temp.id] = temp
        time.sleep(0.01)
#         temp = Ship(np.array([500.0, 50.0]), np.array([0.5, 2.0]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         
#         temp = Ship(np.array([500.0, 480.0]), np.array([0.0, -2.0]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         #  对遇和 左舷交叉相遇     3 无人艇会遇
#         temp = Ship(np.array([200.0, 300.0]), np.array([1.2, -1.1]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         #  四无人艇   会遇
#         temp = Ship(np.array([700.0, 500.0]), np.array([-2.0, -3.0]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         #再来一个追越
#         temp = Ship(np.array([600.0, 20.0]), np.array([-2.0, 1.0]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         # 新加的 6号
#         temp = Ship(np.array([200.0, 450.0]), np.array([-0.1, 2.0]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         # 7
#         temp = Ship(np.array([750.0, 300.0]), np.array([-3.0, 2.]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         # 8
#         temp = Ship(np.array([10.0, 30.0]), np.array([2.0, 2.5]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         # 9
#         temp = Ship(np.array([600.0, 150.0]), np.array([1.0, 1.2]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
#         # 10
#         temp = Ship(np.array([321.0, 20.0]), np.array([-0.1, 1.5]), width=self.window_width, height=self.window_height)
#         self.ships[temp.id] = temp
#         time.sleep(0.01)
        
        for k, v in self.ships.items():
            if self.train_id is None:
                self.train_id = k
            self.all_observations[k] = v.getObservation(self.dis, **self.ships)
        
        #print("本次训练的id号码是:", self.train_id)
        return self.all_observations
        pass

if __name__ == "__main__":
    
    env = Viewer()
    observations = env.reset()
    step = 0
    dic = {}
    while step < 1000:
        c, d, e = env.step(**dic)
        step += 1
        
    
    















