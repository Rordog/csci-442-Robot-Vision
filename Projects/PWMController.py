class PWMController:

    MIN_SERVO_POSITION = 7900
    MAX_SERVO_POSITION = 4100
    CENTER_SERVO_POSITION = (MIN_SERVO_POSITION + MAX_SERVO_POSITION)/2.0

    def __init__(self, channel, usb_controller):
        self.channel = channel
        self.usb_controller = usb_controller
        self.current_state = None

    def set(self, power):
        target = power * (self.MAX_SERVO_POSITION - self.CENTER_SERVO_POSITION) + self.CENTER_SERVO_POSITION
        self.setTarget(target)

    def setTarget(self, target):
        target = int(target)
        self.current_state = target
        lsb = target & 0x7f  # 7 bits for least significant byte
        msb = (target >> 7) & 0x7f  # shift 7 and take next 7 bits for msb
        cmd = chr(0x04) + chr(self.channel) + chr(lsb) + chr(msb)
        self.usb_controller.sendCmd(cmd)

    def reset(self):
        self.setTarget(self.CENTER_SERVO_POSITION)
