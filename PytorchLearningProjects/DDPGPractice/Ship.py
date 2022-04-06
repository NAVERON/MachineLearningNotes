
import numpy as np
import math
import datetime
from collections import deque
import pandas as pd


class Ship():  # 训练对象的属性
    
    K = 0.0785
    T = 3.12
    
    kp=1.2
    ki=20.0
    kd=10.0
    #  标准
    #  位置以左下角为标准原点，这里全部按照实际的坐标系来，绘制的时候再  进一步处理绘制的问题
    #  速度以正上方为 0 ，顺时针旋转    正向
    #  
    def __init__(self, position=np.array([500, 400], dtype=np.float), velocity=np.array([2, 4], dtype=np.float), width=1000.0, height=600.0):  # 矩阵
        self.id = datetime.datetime.now().strftime("%d%H%M%S%f")
        self.isDead = False
        
        self.rudder = 0
        self.position = position
        #self.position[1] = height - self.position[1]   # 左下角为坐标系原点
        self.velocity = velocity
        
        self.width = width
        self.height = height
        self.history = deque(maxlen=40)
        
        self.trajectories = []
        self.setDestination()
        self.now_near = []
        
        # 关于PID的参数设置
        self.dc = 0  # 当前航向与目标航向的偏差
        self.last_course_error = 0
        self.pre_course_error = 0
        
        self.i = 0
        
    def PID_rudder(self, dc):  # dc表示角度偏差，此函数       根据航向偏差修改舵角数值
        self.dc = dc
        delta_rudder = self.kp*(self.dc-self.last_course_error)+(self.kp*self.kd)*(self.dc-2*self.last_course_error+self.pre_course_error)
        self.pre_course_error = self.last_course_error
        self.last_course_error = self.dc
        self.rudder += delta_rudder
        if self.rudder > 35 or self.rudder < -35:
            self.rudder -= delta_rudder
    
    def setDestination(self):
        self.destination = self.position + 500 * self.velocity
        self.destination = np.array([self.destination[0]%self.width, self.destination[1]%self.height])
    def dis_Destination(self):
        return np.linalg.norm(self.destination - self.position)
    
    def courseTurn(self, dc):  # dc代表变化的方向
        # 返回 新的速度矢量，将事例的速度重新设置
        #self.PID_rudder(dc)    # 根据航向偏差啊修改舵角
#         if dc > 0:
#             self.rudderChange(0.2)
#         elif dc < 0:
#             self.rudderChange(-0.2)
        self.rudderChange(dc)
        delta_course = self.K * self.rudder * (1 - self.T + self.T * math.exp(-1./self.T))
        
        dc_radius = np.radians(-delta_course)  # 转换成弧度
        c, s = np.cos(dc_radius), np.sin(dc_radius)
        R = np.array([ [c, -s], [s, c] ])
        self.velocity = np.dot(R, self.velocity.T).T
        #print(self.id, "id:", self.velocity)
    
    def addHistory(self, his):
        if len(self.history) >= 40:
            self.history.popleft()
        self.history.append(his)
    
    def velocityChange(self, dv): # 根据dv  修改原速度矢量   输入的是2维向量S[]
        self.velocity += dv
        if self.getSpeed() > 8 or self.getSpeed() < 0:   # 控制速度大小
            self.velocity -= dv
    def speedChange(self, ds):  # 速度变化过快
        if  self.getSpeed() <= 1 or self.getSpeed() > 5:
            return
        #ds /= 2.0   #速度变化过快
        course = math.radians(self.getCourse())
        sx, sy = ds*math.sin(course), ds*math.cos(course)
        self.velocityChange(np.array([sx, sy]))
    def rudderChange(self, dr):  # 舵角变化    范围为每次一度， 变化舵角会造成航向的变化
        self.rudder += dr
        if self.rudder > 30 or self.rudder < -30:  # 设定舵角的范围
            self.rudder -= dr
    def rudderPositiveOn(self):
        if math.fabs(self.rudder) < 0.5:
            self.rudder = 0
            return
        self.rudder += -self.rudder/5
    
    def getSpeed(self):  # 速度大小
        return np.linalg.norm(self.velocity)
    def getCourse(self): # 运动方向
        return self.calAngle(self.velocity[0], self.velocity[1])
    def distance(self, other_ship):
        return np.linalg.norm(other_ship.position - self.position)
    
    def calAngle(self, dx, dy):   # 计算的角度按照顺时针旋转，正向向上是0度角
        theta = math.atan2(dx, dy)
        angle = math.degrees(theta)
        if angle < 0:
            angle += 360
        
        return angle
    
    def goAhead(self):
        # 边界判断    设置为不能超越边界
        if self.position[0] <= 0:
            self.position[0] = self.width
        elif self.position[0] > self.width:
            self.position[0] = 0
        if self.position[1] <= 0:
            self.position[1] = self.height
        elif self.position[1] > self.height:
            self.position[1] = 0
        # 前面courseTurn改变了舵角和航向，这里直接计算前近
#         delta = self.K * self.rudder * (1 - self.T + self.T * math.exp(-1./self.T))
#         self.courseTurn(delta)
        self.position += self.velocity  # 这样就更新位置了    ====  可以把界面更新放到数据更新里面同步，更好

        self.addHistory([self.position[0], self.position[1]])
#         if self.i%4 == 0:
#             self.trajectories.append([self.position[0], self.position[1], self.getCourse(), self.getSpeed(), self.rudder])
        self.trajectories.append([self.position[0], self.position[1], self.getCourse(), self.getSpeed(), self.rudder])
        self.i += 1
        if np.linalg.norm(self.position-self.destination) < 10:
            self.setDestination()
        #print(self.id + ", rudder : ", self.rudder)
        # 当周边没有无人艇的时候，回航向
        if not self.now_near:
            delta_D = self.destination-self.position
            #delta_angle = self.calAngle(delta_D[0], delta_D[1]) - self.getCourse()  # 偏差航向
            delta_action = np.cross(self.velocity, delta_D)
            #print("全局下的偏差角", delta_angle)
            # 求取偏差航向
            if delta_action > 5:
                #print("需要左转", delta_action)
                self.courseTurn(-5)
            elif delta_action < -5:
                #print("需要右转", delta_action)
                self.courseTurn(5)
            else:
                self.rudderPositiveOn()   #如果没有大偏差就，舵角回正
            #print("当前舵角：", self.rudder)
    
    def storeTrajectories(self):
        formated_data = pd.DataFrame(data = self.trajectories)
        formated_data.to_csv("../"+self.id+".csv", encoding="utf-8", header=None, index=None)
        pass
    def isCollision(self):   # 判断当前是否于与周边碰撞
        for s in self.now_near:
            dis = self.distance(s)
            if dis < 25:  # 如果碰撞，直接返回结果，结束遍历
                self.isDead = True
                s.isDead = True
                
                self.velocity = np.array([0., 0.])
                self.rudder = 0.
                s.velocity = np.array([0., 0.])
                s.rudder = 0.
                
                return True
            # 直接结束遍历
        return False
        
    def getObservation(self, dis, **ships):
        self.now_near = self.getNear(dis, **ships)
        near_locals = self.warpAxis(self.now_near)   # 可以得到   以本艇为中心的环境图
        #print(near_locals[0].toString())   # 测试环境坐标转换是否正确
        dis_of_des = self.dis_Destination()
        observation = [self.getSpeed(), dis_of_des]
        
        up = []
        right = []
        down = []
        left = []
        for local in near_locals:
            ratio = local.local_ratio
            if ratio > 350 and ratio < 10:
                up.append(local)
            elif ratio > 10 and ratio < 90:
                right.append(local)
            elif ratio > 90 and ratio < 247.5:
                down.append(local)
            elif ratio > 247.5 and ratio < 355:
                left.append(local)
        
#         up_average_dis = self.getAverageDis(up)
#         right_average_dis = self.getAverageDis(right)
#         down_average_dis = self.getAverageDis(down)
#         left_average_dis = self.getAverageDis(left)
#         
#         up_average_cou = self.getAverageCourse(up)
#         right_average_cou = self.getAverageCourse(right)
#         down_average_cou = self.getAverageCourse(down)
#         left_average_cou = self.getAverageCourse(left)
#         
#         up_average_ratio = self.getAverageRatio(up)
#         right_average_ratio = self.getAverageRatio(right)
#         down_average_ratio = self.getAverageRatio(down)
#         left_average_ratio = self.getAverageRatio(left)
#         
#         up_average_state = self.getAverageState(up)
#         right_average_state = self.getAverageState(right)
#         down_average_state = self.getAverageState(down)
#         left_average_state = self.getAverageState(left)
#         
#         up_observation = [len(up), up_average_dis, up_average_cou, up_average_ratio, up_average_state]
#         right_observation = [len(right), right_average_dis, right_average_cou, right_average_ratio, right_average_state]
#         down_observation = [len(down), down_average_dis, down_average_cou, down_average_ratio, down_average_state]
#         left_observation = [len(left), left_average_dis, left_average_cou, left_average_ratio, left_average_state]
        
        up_observation = self.getNewObservation(up)
        right_observation = self.getNewObservation(right)
        down_observation = self.getNewObservation(down)
        left_observation = self.getNewObservation(left)
        
        observation.extend(up_observation)
        observation.extend(right_observation)
        observation.extend(down_observation)
        observation.extend(left_observation)
        
        return observation
    def getNewObservation(self, local_ships_list):
        
        dis = 0
        min_ratio = -1
        max_ratio = -1
        v_x = 0
        v_y = 0
        for local_ship in local_ships_list:
            dis += local_ship.local_dis
            this_ratio = local_ship.local_ratio
            if min_ratio < 0 or max_ratio < 0:
                min_ratio = this_ratio
                max_ratio = this_ratio
            else:
                if this_ratio <= min_ratio:
                    min_ratio = this_ratio
                if this_ratio >= max_ratio:
                    max_ratio = this_ratio
            
            velocity = local_ship.getVelocity()
            v_x += velocity[0]
            v_y += velocity[1]-self.velocity[1]
        observation = [dis, min_ratio, max_ratio, v_x, v_y]
        
        return observation
        pass
    def getNear(self, dis, **ships):  # 传入查找对象的引用this_ship，以及距离范围 dis
        self.now_near.clear()   # 清空之前的数据
        
        for v in ships.values():
            item_ship = v
            if self.id == item_ship.id:
                continue
            if self.distance(item_ship) < dis:
                self.now_near.append(item_ship)
        return self.now_near
        pass
    def warpAxis(self, near):
        near_locals = []
        for ship in near:
            d_pos = ship.position - self.position
            d_pos_radius = -np.radians(self.getCourse())  # 转换成弧度
            c, s = np.cos(d_pos_radius), np.sin(d_pos_radius)
            R = np.array([ [c, -s], [s, c] ])
            position = np.dot(d_pos, R)
            
            dh = ship.getCourse() - self.getCourse()    # dh代表坐标转换后的方向
            while dh >= 360 or dh < 0:
                if dh >= 360:
                    dh -= 360
                if dh < 0:
                    dh += 360
            
            ratio = self.calAngle( position[0], position[1] )
            dis = np.linalg.norm(d_pos)
            
            local_ship = LocalShip(ship.id, position, dh, ship.getSpeed(), ratio, dis)
            near_locals.append(local_ship)
        
        return near_locals
    def getAverageDis(self, local_ships_list):
        if not local_ships_list:  # 如果是空的
            return -1
        sum = 0
        for local_ship in local_ships_list:
            sum += local_ship.local_dis
        return sum/len(local_ships_list)
    def getAverageCourse(self, local_ships_list):
        if not local_ships_list:  # 如果是空的
            return -1
        sum = 0
        for local_ship in local_ships_list:
            sum += local_ship.local_course
        return sum/len(local_ships_list)
    def getAverageSpeed(self, local_ships_list):
        pass
    def getAverageRatio(self, local_ships_list):
        if not local_ships_list:  # 如果是空的
            return -1
        sum = 0
        for local_ship in local_ships_list:
            sum += local_ship.local_ratio
        return sum/len(local_ships_list)
    def getAverageState(self, local_ships_list):
        if not local_ships_list:  # 如果是空的
            return -1
        local_one = local_ships_list[0]
        return local_one.getPlus()
        pass
    def toString(self):
        return "id:" + self.id + " , position:" + str(self.position) + " , velocity:" + str(self.velocity) + ", course:" + str(self.getCourse())

class LocalShip():
    
    def __init__(self, warp_id, position, course, speed, ratio, dis):
        self.local_id = warp_id
        self.local_position = position
        self.local_course = course
        self.local_speed = speed
        self.local_ratio = ratio
        self.local_dis = dis
        pass
    
    def getPlus(self):
        radian = np.radians(self.local_course)
        vx, vy = self.local_speed*np.sin(radian), self.local_speed*np.cos(radian)
        velocity = np.array([vx, vy])
        return np.cross(self.local_position, velocity)
    def getVelocity(self):
        radian = np.radians(self.local_course)
        vx, vy = self.local_speed*np.sin(radian), self.local_speed*np.cos(radian)
        velocity = np.array([vx, vy])
        return velocity
    
    def toString(self):
        return self.local_id +":\nposition : "+ str(self.local_position[0]) + ","+str(self.local_position[1]) + "\ncourse" + str(self.local_course) +",\nratio and dis:"+str(self.local_ratio)+"=="+str(self.local_dis)  















