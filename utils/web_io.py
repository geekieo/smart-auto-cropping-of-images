# -*- coding: utf-8 -*-

import re
import six
from urllib.request import urlopen
import numpy as np
import cv2


URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')

def _is_URL(filename):
    """Return True if string is an http or ftp path."""
    return (isinstance(filename, six.string_types) and
            URL_REGEX.match(filename) is not None)

def urlimread(url):
    try:
        if not _is_URL(url):
            raise IOError('Not URL: "%s"'%url)
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    url_list=[
        r'',
        r'http://sakjslka',
        r'http://p0.ifengimg.com/pmop/2018/1114/EE85B48F71465531BABEC68653DD9D4107CF7870_size85_w311_h557.jpeg',
        r'http://p0.ifengimg.com/pmop/2018/1110/66A97991D8FBDD28A1A5D1C8B0FC557C1D1441ED_size56_w457_h686.jpeg',
    ]
    for url in url_list:
        img = urlimread(url)
        if img is not None:
            plt.imshow(img)
            plt.show()
