import os
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc

out_dir = 'out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def _get_file_path(file_name):
    return os.path.join(out_dir, file_name)


def compress(decomposed, k):
    u, s, vh = decomposed
    sigma = list(s[:k])
    sigma.extend([0] * (512-k))
    image = np.dot(u * sigma, vh)
    return image


def plot_distances(k_values, frobenius_distances):
    plot(k_values, frobenius_distances)
    ylabel('frobenius distance')
    xlabel('k')
    title('Distance as a function of k')
    savefig(_get_file_path('Distances.png'))
    clf()


def plot_compression_ratios(k_values, compression_ratios):
    plot(k_values, compression_ratios)
    ylabel('compression ratios')
    xlabel('k')
    title('Compression ratio as a function of k')
    savefig(_get_file_path('CompressionRatios.png'))


def _calculate(image, k_values, image_indeces):
    decomposed = svd(image)
    frobenius_distances = []
    compression_ratios = []
    image_results = []

    for k in k_values:
        compressed_image = compress(decomposed, k)

        compression_ratio = 1 - (k / 512)
        compression_ratios.append(compression_ratio)

        image_difference = compressed_image - image
        frobenius_distance = norm(image_difference)
        frobenius_distances.append(frobenius_distance)

        if k in image_indeces:
            image_result = (compressed_image, k, frobenius_distance, compression_ratio)
            image_results.append(image_result)

    return frobenius_distances, compression_ratios, image_results


def plot_image_results(image_results):

    for image, k, frobenius_distance, compression_ratio in image_results:
        image_file_name = 'image.{}.png'.format(k)
        imsave(_get_file_path(image_file_name), image)

        results_file_name = 'results.{}.txt'.format(k)
        with open(_get_file_path(results_file_name), 'w') as file_:
            file_.write('K = {}\n'.format(k))
            file_.write('Compression ratio = {}\n'.format(compression_ratio))
            file_.write('Frobenius distance = {}\n'.format(frobenius_distance))


if __name__ == '__main__':

    image = misc.ascent()

    k_values = range(0, 512)
    image_indeces = [50, 100, 200, 300, 500]

    frobenius_distances,  compression_ratios, image_results =  _calculate(image, k_values, image_indeces)

    plot_distances(k_values, frobenius_distances)
    plot_compression_ratios(k_values, compression_ratios)
    plot_image_results(image_results)

