from serial import Serial
import sys

class USBController:

    def __init__(self):
        try:
            self.usb = Serial('/dev/ttyACM0')
        except:
            try:
                pass
                #self.usb = Serial('/dev/ttyACM1')
            except:
                print("No servo serial ports found")
                self.usb = None
    def sendCmd(self, cmd):
        if self.usb is not None:
            cmdStr = chr(0xaa) + chr(0x0c) + cmd
            self.usb.write(bytes(cmdStr, 'latin-1'))
            print("Sending command: " + cmdStr)
        else:
            print("Fake sending command")
