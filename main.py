import click
import os, shutil
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
            arr.append(color)
        return arr

def cleanSequence():
    for filename in os.listdir('sequence'):
        file_path = os.path.join('sequence', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def cleanSamples():
    for filename in os.listdir('samples'):
        file_path = os.path.join('samples', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



@click.command()
@click.option('--input', required=True, help='The input video path (supports all that ffmpeg does)', type=click.Path(exists=True))
@click.option('--output', default='./poster', help='The output path (.PNG only)', type=click.Path(writable=True))
@click.option('--seconds-per-frame', default=300, help='How to slice the video - 60 would be 1 minute between frames', type=int)
@click.option('--accuracy', default=128, help='The size we compress the images to when sampling (in pixels)', type=int)
@click.option('--scale', default=1, help='The scale of the setup (in pixels)', type=int)
@click.option('--palette-size', default=5, help='How many bands (palettes) to draw from the images', type=int)
@click.option('--focus', default=False, help='Which band (1 to palette-size) to focus on', type=int)

def main(input, output, seconds_per_frame, accuracy, scale, palette_size, focus):
    """Simple program that greets NAME for a total of COUNT times."""
    cleanSequence()
    cleanSamples()

    os.system('./lib/ffmpeg -i ' + input + ' -r ' + str(1/seconds_per_frame) + ' ./sequence/raw_%04d.png')

    frames = len(os.listdir('sequence'))
    band_width = 2 * scale
    band_height = 200 * scale


    poster = Image.new('RGB', (band_width * frames, band_height), color = 'white')

    draw = ImageDraw.Draw(poster)


    for num in range(0, frames):
        num_pad = str(num + 1).rjust(4, '0')
        image = Image.open('sequence/raw_' + num_pad + '.png')
        new_image = image.resize((accuracy, accuracy))

        samp = 'samples/'+ num_pad +'.png'

        new_image.save(samp)
        image = cv2.imread(samp)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reshape = image.reshape((image.shape[0] * image.shape[1], 3))

        cluster = KMeans(n_clusters=palette_size).fit(reshape)
        arr = visualize_colors(cluster, cluster.cluster_centers_)

        if len(arr) < palette_size:
            arr = np.pad(arr, palette_size - len(arr), 'edge').tolist()

        color_height = band_height / palette_size

        if (isinstance(focus, int)):
            color = arr[focus]
            draw.rectangle([num * band_width, 0, (num + 1) * band_width, band_height], fill=(floor(color[0]), floor(color[1]), floor(color[2]), 255))
        else:
            for i, color in enumerate(arr):
                draw.rectangle([num * band_width, i * color_height, (num + 1) * band_width, (i + 1) * color_height], fill=(floor(color[0]), floor(color[1]), floor(color[2]), 255))
    del draw

    poster.save(click.format_filename(output) + '.png', "PNG")

if __name__ == '__main__':
    main()
