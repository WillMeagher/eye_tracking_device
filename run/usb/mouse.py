LOGICAL_MIN = 1
LOGICAL_MAX = 4095

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def clip(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def write_report(report):
    with open('/dev/hidg0', 'wb+') as fd:
        fd.write(report)

def move(x, y, click=0):
    # convert to pixels
    x = int(x * (LOGICAL_MAX / SCREEN_WIDTH))
    y = int(y * (LOGICAL_MAX / SCREEN_HEIGHT))

    # normalize to value on screen
    x = clip(x, LOGICAL_MIN, LOGICAL_MAX)
    y = clip(y, LOGICAL_MIN, LOGICAL_MAX)

    # convert to hex values
    x_high = "{:02x}".format(x >> 8)
    x_low = "{:02x}".format(x & 0xff)

    y_high = "{:02x}".format(y >> 8)
    y_low = "{:02x}".format(y & 0xff)

    cursor = "{:02x}".format(click)

    # combine values and convert to bytes
    report_val = int((cursor + x_low + x_high + y_low + y_high), 16)
    report = report_val.to_bytes(5, 'big')

    # output number
    write_report(report)

def main():
    for i in range(100, 1000):
        move(i, i)
    return

if __name__ == '__main__':
    main()
