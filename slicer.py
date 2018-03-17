import numpy as np
import cv2
import progressbar
import argparse
import os


def slice_image(img, num_slices_per_axis):
	slice_shape = (img.shape[0] / num_slices_per_axis, img.shape[1] / num_slices_per_axis)
	for i in range(num_slices_per_axis):
		for j in range(num_slices_per_axis):
			top_left = (i * slice_shape[0], j * slice_shape[1])
			yield img[top_left[0]:(top_left[0]+slice_shape[0]), top_left[1]:(top_left[1]+slice_shape[1])]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dataset image slicer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('dataset_dir_path', help="Path to directory that contains subdirectories of images")
	parser.add_argument('--output_dir_path', '-o', default="output", help="Path to output directory")
	parser.add_argument('--num_slices_per_axis', '-n', type=int, default=3, help="Number of slices per sides. Total slices = num_slices_per_axis ^ 2")	
	args = parser.parse_args()
	mypath = args.dataset_dir_path
	outpath = args.output_dir_path
	num_slices_per_axis = args.num_slices_per_axis

	# preare output dir
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	count = 0
	num_dir = len([name for name in os.listdir(mypath)])
	bar = progressbar.ProgressBar(maxval=num_dir).start()
	for f in os.listdir(mypath):
		path = os.path.join(mypath, f)
		for f_sub in os.listdir(path):
				path_sub = os.path.join(path, f_sub)                
				if os.path.isfile(path_sub):
					try:
						img = cv2.imread(path_sub)
						slice_count = 1
						for sliced in slice_image(img, num_slices_per_axis):
							outpath_sub = os.path.join(outpath, f)
							if not os.path.exists(outpath_sub):
								os.makedirs(outpath_sub)
							basename, ext = os.path.splitext(f_sub)
							sliced_name = "{}_{}{}".format(basename, slice_count, ext)
							cv2.imwrite(os.path.join(outpath_sub, sliced_name), sliced)
							slice_count = slice_count +1

					except Exception as err:
						print(err)
						print(path_sub)
						sys.exit(0)
		count += 1
		bar.update(count)
	bar.finish()

