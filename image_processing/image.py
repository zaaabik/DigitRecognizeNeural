import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage.measurements import center_of_mass
from skimage import io


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def resize_image(path):
    raw_image = Image.open(path)
    image = raw_image.resize((24, 24))
    image.save('test2.png')


def open_image(path):
    image = Image.open(path)
    if image.size != (20, 20):
        image = image.resize((20, 20))
    image = ImageOps.invert(image)
    gray = image.convert('L')
    gray = np.array(gray)
    gray[gray < 128] = 0
    gray[gray >= 128] = 1
    gray = np.pad(gray, pad_width=4, mode='constant', constant_values=0)
    tmp = Image.fromarray(gray * 255)
    gray = gray.flatten()
    tmp = ImageOps.invert(tmp)
    tmp.save('tmp.bmp')
    return gray


def create_image(gray_array, shape, path):
    gray_array = np.reshape(gray_array, shape)
    created_image = Image.fromarray(gray_array * 255)
    created_image = created_image.convert('RGB')
    img = np.array(created_image)
    img = img[:, :, ::-1].copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    print(w, h)
    print(x, y)
    created_image.save(path)


def get_features_from_array(array):
    img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.destroyAllWindows()
    ar = np.array(thresh)
    ar = ar[y:y + h, x:x + w]
    im = Image.fromarray(ar).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    # img = cv2.resize(ar, (20, 20), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ar = np.array(img)
    ar = np.pad(ar, pad_width=4, mode='constant', constant_values=0)
    img = Image.fromarray(ar)
    img.save('croped.bmp')
    imagesums = np.sum(np.sum(ar, axis=1), axis=1)
    indices = np.arange(28)
    X, Y = np.meshgrid(indices, indices)
    centroidx = np.sum(ar * X) / imagesums;
    centroidy = np.sum(ar * y) / imagesums;

    # What range do centroid coordinates span?

    print(centroidx)
    print(centroidy)

    return np.array(img).flatten() / 255


def load_image_from_url(url):
    img = io.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('t', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.destroyAllWindows()
    ar = np.array(thresh)
    ar = ar[y:y + h, x:x + w]
    img = cv2.resize(ar, (20, 20), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ar = np.array(img)
    ar = np.pad(ar, pad_width=4, mode='constant', constant_values=0)
    img = Image.fromarray(ar)
    img.save('croped.bmp')
    return np.array(img).flatten() / 255


def load_image_from_array(array):
    img = cv2.imdecode(array, 0)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.destroyAllWindows()
    ar = np.array(thresh)
    ar = ar[y:y + h, x:x + w]
    img = cv2.resize(ar, (20, 20), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ar = np.array(img)
    ar = np.pad(ar, pad_width=4, mode='constant', constant_values=0)
    c = center_of_mass(ar)
    dx = int(round(14 - c[1]))
    dy = int(round(14 - c[0]))
    ar = np.roll(ar, dy, axis=0)
    ar = np.roll(ar, dx, axis=1)
    img = Image.fromarray(ar)
    img.save('croped.bmp')
    return np.array(img).flatten() / 255


def load_image(path):
    img = cv2.imread(path, 0)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    ar = np.array(thresh)
    ar = ar[y:y + h, x:x + w]
    im = Image.fromarray(ar).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    # img = cv2.resize(ar, (20, 20), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ar = np.array(newImage)
    c = center_of_mass(ar)
    print(c)
    dx = 14 - int(c[1])
    dy = 14 - int(c[0])
    ar = np.roll(ar, dy, axis=0)
    ar = np.roll(ar, dx, axis=1)
    img = Image.fromarray(ar)
    img.save('croped.bmp')
    return ar.flatten() / 255


if __name__ == '__main__':
    load_image('test.bmp')
