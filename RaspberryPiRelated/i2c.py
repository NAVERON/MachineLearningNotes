


import smbus
import time
import math

bus = smbus.SMBus(1)
address = 0x0d
MagnetcDeclination = -2.54
CalThreshold = 0
flag = True


def init():
    bus.write_byte_data(address, 0x0b, 0x01)
    bus.write_byte_data(address, 0x09, 0x1d)
    bus.write_byte_data(address, 0x20, 0x40)
    bus.write_byte_data(address, 0x21, 0x01)
    offsetX, offsetY, offsetZ, xgain, ygain = calibrateMag()
    return offsetX, offsetY, offsetZ, xgain, ygain


def getRawData():
    try:
        status = bus.read_byte_data(address, 0x06)
        if (status & 0x01 == 0x01 or status & 0x04 == 0x04) and (status & 0x02 != 0x02):
            x = (bus.read_byte_data(address, 0x01) << 8) | (bus.read_byte_data(address, 0x00))
            y = (bus.read_byte_data(address, 0x03) << 8) | (bus.read_byte_data(address, 0x02))
            z = (bus.read_byte_data(address, 0x05) << 8) | (bus.read_byte_data(address, 0x04))
            return x, y, z
    except Exception:
        return None, None, None
    return None, None, None


def getHeading(x, y, z, offsetX, offsetY, offsetZ, xgain, ygain):
    headingRadians = math.atan2(ygain * (y - offsetY), xgain * (x - offsetX))

    headingDegrees = headingRadians * 180 / math.pi
    headingDegrees = headingDegrees % 360
    headingRadians += MagnetcDeclination
    if headingDegrees > 360:
        headingDegrees -= 360
    elif headingDegrees < 0:
        headingDegrees += 360
    return headingDegrees


def calibrateMag():
    x, y, z = getRawData()
    while x is None:
        x, y, z = getRawData()
    xMax = xMin = x
    yMax = yMin = y
    zMax = zMin = z
    offsetX = offsetY = offsetZ = 0
    for i in range(0, 200):
        x, y, z = getRawData()
        while x is None:
            x, y, z = getRawData()
        if x > xMax:
            xMax = x
        elif x < xMin:
            xMin = x
        if y > yMax:
            yMax = y
        elif y < yMin:
            yMin = y
        if z > zMax:
            zMax = z
        elif z < zMin:
            zMin = z
        time.sleep(0.1)
    if abs(xMax - xMin) > CalThreshold:
        offsetX = (xMax + xMin) / 2
    if abs(zMax - zMin) > CalThreshold:
        offsetZ = (zMax + zMin) / 2
    if abs(yMax - yMin) > CalThreshold:
        offsetY = (yMax + yMin) / 2
    xgain = 1
    ygain = (xMax - xMin) / (yMax - yMin)
    return offsetX, offsetY, offsetZ, xgain, ygain


def i2c(offsetX, offsetY, offsetZ, xgain, ygain):
    while flag:
        x, y, z = getRawData()
        while x is None:
            x, y, z = getRawData()
        heading = getHeading(x, y, z, offsetX, offsetY, offsetZ, xgain, ygain)
        print(heading)
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        offsetX, offsetY, offsetZ, xgain, ygain = init()
        i2c(offsetX, offsetY, offsetZ, xgain, ygain)
    except KeyboardInterrupt:
        flag = False
