import argparse
import os
import glob
import time

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import moviepy.editor as mpy

from common import rle_decode, process_img

def get_color_dict():
    """
    Return a color palette for visualization
    Returns:
        Dictionary of colors
    """
    return {
        'large_bowel' : (255/255., 178/255., 102/255.),
        'small_bowel' : (102/255., 255/255., 255/255.),
        'stomach' : (178/255., 102/255., 255/255.)
    }

def main(args):
    # load mask file
    df = pd.read_csv(args.csv_file)

    # list all slice image files
    case_id = args.case_id
    case, day = case_id.split('_')
    case_folder = os.path.join(args.img_folder, case, case_id, "scans")
    if not os.path.exists(case_folder):
        print("{:s} not found in the dataset.".format(case_id))
        return

    slice_file_list = sorted(glob.glob(os.path.join(case_folder, '*.png')))
    num_slices = len(slice_file_list)

    # parse slice_id, image resolution, and spacing from file name
    tokens = os.path.basename(slice_file_list[0])[:-4].split('_')
    slice_id = '_'.join(tokens[:2])
    im_w, im_h = int(tokens[2]), int(tokens[3])
    w_spacing, h_spacing, z_spacing = float(tokens[4]), float(tokens[5]), 3.

    # prepare for outputs
    mask_dict = {
        "small_bowel": np.zeros([num_slices, im_h, im_w], dtype=bool),
        "large_bowel": np.zeros([num_slices, im_h, im_w], dtype=bool),
        "stomach": np.zeros([num_slices, im_h, im_w], dtype=bool)
    }
    img_data = np.zeros([num_slices, im_h, im_w])

    # loop over all slices and load data into tensors
    start_time = time.time()
    case_df = df.loc[df["id"].str.contains(case_id)]
    for slice_idx, slice_file in enumerate(slice_file_list):
        # get slice_id
        slice_id  = '_'.join(os.path.basename(slice_file).split('_')[:2])

        # load imaging data
        img_data[slice_idx, :, :] = np.asarray(Image.open(slice_file))

        # load mask data
        slice_mask = case_df.loc[case_df['id'] == (case_id + '_' + slice_id)]
        if "predicted" in slice_mask.columns:
            col_name = "predicted"
        elif "segmentation" in slice_mask.columns:
            col_name = "segmentation"
        else:
            print("Mask not found in the csv file.")
            return

        for row_idx, row in slice_mask.iterrows():
            if (not pd.isnull(row[col_name])) and row['class'] in mask_dict:
                cur_mask = rle_decode(row[col_name], (im_h, im_w), dtype=bool)
                mask_dict[row['class']][slice_idx, :, :] = cur_mask

    # log the information
    end_time = time.time()
    print("[{:s}] Loaded {:d} slices of resolution {:d} (H) x {:d} (W) in {:0.2f} sec.".format(
            case_id, num_slices, im_h, im_w, end_time - start_time
        )
    )

    # visualize slices and masks with an animated GIF
    if args.viz:
        os.makedirs("./outputs", exist_ok=True)
        color_dict = get_color_dict()
        alpha = args.alpha
        fps = args.fps
        font = ImageFont.truetype("FreeMono.ttf", 50)
        w_offset, h_offset = 40, 40

        viz_img_list = []
        for idx, img in enumerate(img_data):
            # contrast enhancement / normalization -> convert to color image
            cur_img = process_img(img)
            cur_img = np.tile(cur_img[:, :, np.newaxis], (1, 1, 3))

            # loop over all organ types
            for type in color_dict.keys():
                cur_color = np.array(color_dict[type]).reshape(1, 1, 3)
                color_map = np.tile(cur_color, (im_h, im_w, 1))

                cur_mask = mask_dict[type][idx, :, :].astype(np.float32)
                cur_mask = np.tile(cur_mask[:, :, np.newaxis], (1, 1, 3))
                # alpha blending
                cur_img += ((alpha-1) * cur_img + (1-alpha) * color_map) * cur_mask

            # clipping and type casting
            cur_img = np.clip(255 * cur_img, 0, 255).astype(np.uint8)

            # copy to PIL image for drawing the legends
            viz_img = np.zeros(
                [cur_img.shape[0], 2*cur_img.shape[1], cur_img.shape[2]],
                dtype=np.uint8
            )
            viz_img[:, cur_img.shape[1]:, :] = cur_img
            viz_img = Image.fromarray(viz_img)
            # upsample the image by 2x for better visualization
            viz_img = viz_img.resize((4*cur_img.shape[1], 2*cur_img.shape[0]))

            # adding legends
            draw = ImageDraw.Draw(viz_img)
            margin = 2 * (cur_img.shape[0] - h_offset) / len(color_dict.keys())
            for t_idx, type in enumerate(color_dict.keys()):
                draw.text(
                    (w_offset, t_idx * margin + h_offset),
                    type,
                    font=font,
                    fill=tuple([int(255*c) for c in color_dict[type]])
                )

            viz_img = np.asarray(viz_img)
            viz_img_list.append(viz_img)

        clip = mpy.ImageSequenceClip(viz_img_list, fps=fps)
        clip.write_gif(
            os.path.join("./outputs", "{:s}.gif".format(case_id)),
            fps=fps
        )


################################################################################
if __name__ == '__main__':
    """Sample code for loading and visualizing MRI data and their masks"""
    # the arg parser
    parser = argparse.ArgumentParser(description='Sample code for data loading')
    parser.add_argument(
        'csv_file',
        type=str,
        metavar='DIR',
        help='Path to a CSV file with mask.'\
             'This can be either the ground truth annotations, '\
             'or the predicted results.'
    )
    parser.add_argument(
        'img_folder',
        type=str,
        metavar='DIR',
        help='Path to the image folder.'
    )
    parser.add_argument(
        'case_id',
        type=str,
        help='Case ID that specifies the scan to be loaded.'
    )
    parser.add_argument(
        '-alpha',
        type=float,
        default=0.7,
        help='Alpha values used for blending during visualization'
    )
    parser.add_argument(
        '-fps',
        type=int,
        default=15,
        help='FPS used for animation during visualization'
    )
    parser.add_argument(
        '-viz',
        action='store_true',
        help='If to visualize slices and maks (Defalut: False).'\
             'Visualization will be saved under ./outputs.'
    )
    args = parser.parse_args()
    main(args)
