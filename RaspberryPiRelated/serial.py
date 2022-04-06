import serial
import time

ser = serial.Serial("/dev/ttyAMA0",9600)


def Serial():
    while True:
        count = ser.inWaiting()
        if count != 0:
            recv = ser.read(count)
            ser.write(recv)
        ser.flushInput()
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        Serial()
    except KeyboardInterrupt:
        if ser != None:
            ser.close()