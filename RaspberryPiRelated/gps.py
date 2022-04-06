# coding=<utf8>


import serial
import time

ser = serial.Serial("/dev/ttyAMA0", 9600)

Payload = {'iTOW' : 0,
               'lon' : 0,
               'lat' : 0,
               'height' : 0,
               'hMSL' : 0,
               'hAcc' : 0,
               'vAcc' : 0}
CK_A = 0
CK_B = 0
UBX_NAV_VELNED = {'Header' : [0xB5, 0x62],
               'Class' : 0x01,
               'ID' : 0x12,
               'Length' : 36,
               'Payload' : Payload,
               'Checksum':[CK_A, CK_B]}

UBX_NAV_SOL = {'Header' : [0xB5, 0x62],
               'Class' : 0x01,
               'ID' : 0x06,
               'Length' : 52,
               'Payload' : Payload,
               'Checksum':[CK_A, CK_B]}

numCh = 0
UBX_NAV_SVINFO = {'Header' : [0xB5, 0x62],
               'Class' : 0x01,
               'ID' : 0x30,
               'Length' : 8 + 12 * numCh,
               'Payload' : Payload,
               'Checksum':[CK_A, CK_B]}

UBX_NAV_STATUS = {'Header' : [0xB5, 0x62],
               'Class' : 0x01,
               'ID' : 0x03,
               'Length' : 16,
               'Payload' : Payload,
               'Checksum':[CK_A, CK_B]}

UBX_NAV_POSLLH = {'Header' : [0xB5, 0x62],
               'Class' : 0x01,
               'ID' : 0x02,
               'Length' : 28,
               'Payload' : Payload,
               'Checksum':[CK_A, CK_B]}


def getData():
    count = ser.inWaiting()
    if count != 0:
        data = ser.read(911)  ####################################################################各帧不按顺序发送？
        if parseData(data) == 0:
            print("parseData failed!")
    else:
        data = getData()
    return data


def parseData(data):
    data = map(ord, data)
    indexs = []
    i = 0
    flag = 0
    start = 0
    end = 0
    for byte in data:
        if byte == 0xB5:
            indexs.append(i)
        i += 1
    if indexs is None:
        return 0
    for index in indexs:
        if index + 24 < len(data):
            if data[index + 1] == 0x62:
                # find start frame and end frame
                if data[index + 3] == 0x12:
                    if flag == 0:
                        flag += 1
                        start = index
                    elif flag == 1:
                        end = index
                        flag += 1

    j = 0
    # get 5 complete frames
    for index in indexs:
        if index < start or index >= end:
            indexs.pop(j)
        j += 1

    for index in indexs:
        if index + 24 > len(data):
            return 0
        if data[index + 3] == 0x12:
            print("UBX_NAV_VELNED")
        elif data[index + 3] == 0x06:
            print("UBX_NAV_SOL")
        elif data[index + 3] == 0x30:
            print("UBX_NAV_SVINFO")
        elif data[index + 3] == 0x03:
            print("UBX_NAV_STATUS")
        elif data[index + 3] == 0x02:
            print("UBX_NAV_POSLLH")
            UBX_NAV_POSLLH['Payload']['lon'] = ((data[index + 8] << 24)
                                                + (data[index + 7] << 16)
                                                + (data[index + 6] << 8)
                                                + (data[index + 5])) / 10000000 * 3600
            UBX_NAV_POSLLH['Payload']['lat'] = ((data[index + 12] << 24)
                                                + (data[index + 11] << 16)
                                                + (data[index + 10] << 8)
                                                + (data[index + 9])) / 10000000 * 3600
    return 1


def getGPS():
    while True:
        getData()
        print("lon={},lat={}".format(UBX_NAV_POSLLH['Payload']['lon'], UBX_NAV_POSLLH['Payload']['lat']))
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        getGPS()
    except KeyboardInterrupt:
        if ser != None:
            ser.close()
