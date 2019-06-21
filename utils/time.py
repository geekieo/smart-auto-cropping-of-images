import time

def seconds2time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    times = "{:0>2d}h{:0>2d}m{:0>2f}s".format(int(h), int(m), s)
    return times

if __name__ == "__main__":
    timeStamp = time.time()
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    time.sleep(2)

    times = time.time()- timeStamp
    print(seconds2time(times))
    times = times/3
    print(seconds2time(times))