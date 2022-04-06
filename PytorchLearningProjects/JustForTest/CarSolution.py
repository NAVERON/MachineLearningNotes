
import numpy as np
import pyglet
import time


pyglet.clock.set_fps_limit(10000)


class CarEnv(object):
    n_sensor = 5  # 5个探测方向
    action_dim = 1
    state_dim = n_sensor
    viewer = None
    viewer_xy = (1000, 600)    #窗口宽高
    sensor_max = 50.   # 相当于探测距离
    start_point = [500, 300]   #小车初始位置
    speed = 50.
    dt = 0.1
    
    def __init__(self, discrete_action=False):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]
        else:
            self.action_bound = [-1, 1]
        
        self.terminal = False
        # node1 (x, y, r, w, l),
        self.car_info = np.array([0, 0, 0, 10, 20], dtype=np.float64)   # car coordination  // 前两个位置xy，3: 角度， 最后 长宽
        self.obstacle_list = []
        for _ in range(15):
            x = np.random.choice(list(range(900)))
            y = np.random.choice(list(range(500)))
            temp_coords = np.array([   #障碍物位置################################################################
                [x, y],
                [x+80, y],
                [x+80, y+80],
                [x, y+80],
            ])
            self.obstacle_list.append(temp_coords)
        
        self.sensor_info = self.sensor_max + np.zeros((self.n_sensor, 3))  # n sensors, (distance, end_x, end_y)
    
    def step(self, action):
        if self.is_discrete_action:  # 是不是离散动作？
            action = self.actions[action]   # 只能让小车向左或者向右转向
        else:
            action = np.clip(action, *self.action_bound)[0]
        self.car_info[2] += action * np.pi/30  # max r = 6 degree  ===  每次动作向左或向右变化6度
        self.car_info[:2] = self.car_info[:2] + self.speed * self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])
        
        self._update_sensor()
        s = self._get_state()
        r = -1 if self.terminal else 0
        time.sleep(0.001)
        return s, r, self.terminal
        
    def reset(self):
        self.terminal = False
        self.car_info[:3] = np.array([*self.start_point, -np.pi])
        self._update_sensor()
        return self._get_state()

    def render(self):    ############################################################################################
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.obstacle_list)
        self.viewer.render()

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a
    
    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)
    
    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s

    def _update_sensor(self):
        cx, cy, rotation = self.car_info[:3]

        n_sensors = len(self.sensor_info)
        sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, n_sensors)
        xs = cx + (np.zeros((n_sensors, ))+self.sensor_max) * np.cos(sensor_theta)
        ys = cy + (np.zeros((n_sensors, ))+self.sensor_max) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])    # shape (5 sensors, 2)
        
        # sensors
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # rotated x y
        self.sensor_info[:, -2:] = np.vstack([rotated_x+cx, rotated_y+cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info)):
            s = self.sensor_info[si, -2:] - q
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[si, -2:]]

            # obstacle collision
            for nano in self.obstacle_list:
                for oi in range(len(nano)):    #########################################################
                    p = nano[oi]
                    r = nano[(oi + 1) % len(nano)] - nano[oi]
                    if np.cross(r, s) != 0:  # may collision
                        t = np.cross((q - p), s) / np.cross(r, s)
                        u = np.cross((q - p), r) / np.cross(r, s)
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            intersection = q + u * s
                            possible_intersections.append(intersection)
                            possible_sensor_distance.append(np.linalg.norm(u*s))
            
            # window collision
            win_coord = np.array([
                [0, 0],
                [self.viewer_xy[0], 0],
                [*self.viewer_xy],
                [0, self.viewer_xy[1]],
                [0, 0],
            ])
            for yona in self.obstacle_list:
                for oi in range(len(yona)):
                    p = win_coord[oi]
                    r = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
                    if np.cross(r, s) != 0:  # may collision
                        t = np.cross((q - p), s) / np.cross(r, s)
                        u = np.cross((q - p), r) / np.cross(r, s)
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            intersection = p + t * r
                            possible_intersections.append(intersection)
                            possible_sensor_distance.append(np.linalg.norm(intersection - q))

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]
            if distance < self.car_info[-1]/2:
                self.terminal = True


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, car_info, sensor_info, obstacle_list):
        super(Viewer, self).__init__(width, height, resizable=False, caption='2D car', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.car_info = car_info
        self.sensor_info = sensor_info

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        self.sensors = []
        line_coord = [0, 0] * 2
        c = (73, 73, 73) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ('v2f', line_coord), ('c3B', c)))

        car_box = [0, 0] * 4
        c = (249, 86, 86) * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ('v2f', car_box), ('c3B', c))

        c = (134, 181, 244) * 4
        for item in obstacle_list:
            self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', item.flatten()), ('c3B', c))
        # self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', obstacle_coords.flatten()), ('c3B', c))

    def render(self):
        pyglet.clock.tick()
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update(self):
        cx, cy, r, w, l = self.car_info
        
        # sensors
        for i, sensor in enumerate(self.sensors):
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]

        # car
        xys = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys = []
        for x, y in xys:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        self.car.vertices = r_xys


if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(30)
    for ep in range(20):
        s = env.reset()
        # for t in range(100):
        while True:
            env.render()
            s, r, done = env.step(env.sample_action())
            if done:
                break
























