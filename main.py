from math import floor
from PIL import Image, ImageDraw
import cv2, numpy as np
from sklearn.cluster import KMeans

def visualize_colors(cluster, centroids, rect=np.zeros((50, 300, 3), dtype=np.uint8)):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    colors = sorted(zip(hist, centroids), key=lambda x: x[0])
    start = 0
    arr = []
    for (percent, color) in colors:
        # print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
        arr.append(color)
    return rect, arr


window = np.zeros((50, 300, 3), dtype=np.uint8)

frames = 25
band_width = 40
band_height = 200
palette_width = 5


poster = Image.new('RGB', (band_width * frames, band_height), color = 'white')

draw = ImageDraw.Draw(poster)


for num in range(0, frames):
    num_pad = str(num + 1).rjust(4, '0')
    image = Image.open('sequence/raw_' + num_pad + '.png')
    new_image = image.resize((128, 128))

    samp = 'samples/'+ num_pad +'.png'

    new_image.save(samp)
    image = cv2.imread(samp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    cluster = KMeans(n_clusters=palette_width).fit(reshape)
    visualize, arr = visualize_colors(cluster, cluster.cluster_centers_, window)

    if len(arr) < palette_width:
        arr = np.pad(arr, palette_width - len(arr), 'edge').tolist()

    major_color = arr[0]

    color_height = band_height / palette_width
    # print(major_color)

    for i, color in enumerate(arr):
        draw.rectangle([num * band_width, i * color_height, (num + 1) * band_width, (i + 1) * color_height], fill=(floor(color[0]), floor(color[1]), floor(color[2]), 255))


print(arr)

del draw

poster.save('poster.png', "PNG")
