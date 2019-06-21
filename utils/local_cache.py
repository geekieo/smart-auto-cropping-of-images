# -*- coding:utf-8 -*-
import time
import weakref
import collections


class LocalCache:

    notFound = object()

    # list dict 等不支持弱引用，但其子类支持，故这里包装了下
    class Dict(dict):
        def __del__(self):
            pass

    def __init__(self, maxlen=10):
        self.weak = weakref.WeakValueDictionary()   # 不是 WeakKeyDictionary
        self.strong = collections.deque(maxlen=maxlen)

    @staticmethod
    def nowTime():
        return int(time.time())

    def get(self, key):
        value = self.weak.get(key, self.notFound)
        if value is not self.notFound:
            expire = value[r'expire']
            if self.nowTime() > expire:
                return self.notFound
            else:
                return value
        else:
            return self.notFound

    def set(self, key, value):
        # strongRef 作为强引用避免被回收
        self.weak[key] = strongRef = LocalCache.Dict(value)
        # 放入队列，弹出元素马上回收
        self.strong.append(strongRef)


if __name__ == "__main__":
    caches = LocalCache(maxlen=10)
    for i in range(10):
        caches.set(key=i, value={i: i, 'expire': caches.nowTime()+60})  # 60秒
    caches.set(key=0, value={0: 0, 'expire': caches.nowTime()+60})
    ret = caches.get(key=0)
    print(ret)
