

# 导入gpio的模块
import gpio as GPIO
import time


class motor:

    def __init__(self):
        # set the numbering method of gpio
        # GPIO.BOARD->board code
        # GPIO.BCM->BCM code
        # GPIO.wiringPi->wiringPi code
        GPIO.setmode(GPIO.BOARD)

        # define the gpio pin of signals
        self.IN1 = 11
        self.IN2 = 12
        self.IN3 = 13
        self.IN4 = 15
        # Servo pwm signal
        self.sPin = 16
        # Motor pwm signal
        self.mPin = 18

        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        GPIO.setup(self.sPin, GPIO.OUT, initial=False)
        GPIO.setup(self.mPin, GPIO.OUT, initial=False)
        # 50Hz
        self.sPwm = GPIO.PWM(self.sPin, 50)
        self.sPwm.start(0)
        self.change_angle(7.5)
        # 50Hz
        self.mPwm = GPIO.PWM(self.mPin, 50)
        # 占空比为0
        self.mPwm.start(0)
        self.change_angle(50)

    # speed[0,100]
    def change_speed(self, speed):
        self.mPwm.ChangeDutyCycle(speed)

    def change_angle(self, angle):
        self.sPwm.ChangeDutyCycle(angle)

    def forward(self, sleep_time):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        # 1.5ms,占空比为7.5
        self.change_angle(7.5)
        time.sleep(sleep_time)
        GPIO.cleanup()

    def left(self, sleep_time):
        GPIO.output(self.IN1, False)
        GPIO.output(self.IN2, False)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        # 0.5ms,占空比为2.5
        self.change_angle(2.5)
        time.sleep(sleep_time)
        GPIO.cleanup()

    def right(self, sleep_time):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, False)
        GPIO.output(self.IN4, False)
        # 2.5ms,占空比为12.5
        self.change_angle(12.5)
        time.sleep(sleep_time)
        GPIO.cleanup()

    def stop(self, sleep_time):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.HIGH)
        # 1.5ms,占空比为7.5
        self.change_angle(7.5)
        time.sleep(sleep_time)
        GPIO.cleanup()





