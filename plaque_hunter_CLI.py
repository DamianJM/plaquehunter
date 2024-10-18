# Inference of phage plaques using plaque and petri models

# imports
from ultralytics import YOLO
from PIL import Image
import cv2, math
import numpy as np
import itertools
from functools import partial, cmp_to_key
import argparse

# Functions

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--study', required=True, default="detect")
    parser.add_argument('--tilesize', required=True)
    parser.add_argument('--tiledim', required=True)
    parser.add_argument('--fast', default=False)
    parser.add_argument('--output', required=False, default="Output.png")
    args = parser.parse_args()
    return args
    
def tile(image, tile_size: int, offset: int, tiledim: int, imageout=True):
    img = cv2.resize(image, ((tile_size[0]*tiledim), (tile_size[1] * tiledim)))
    img_shape = img.shape
    output_images: list = []
    output_image_names: list = []

    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            cropped_img = img[offset[1] * i:min(offset[1] * i+tile_size[1], img_shape[0]), offset[0] * j:min(offset[0]*j+tile_size[0], img_shape[1])]
            output_image_names.append("tile_" + str(i) + "_" + str(j) + ".png")
            output_images.append(cropped_img)
            # output the tiles
            if imageout:
                cv2.imwrite("tile_" + str(i) + "_" + str(j) + ".png", cropped_img)
    
    return output_images, output_image_names

def stitch_tiles(arr, dim: list):
    """function assumes image array in order!"""
    if len(dim) != 2:
        raise IOError("Two dimensional image vector required!")
    if dim[0]*dim[1] != len(arr):
        raise IOError("Invalid dimensions for input tile size!")
    if dim[0] != dim[1]:
        raise IOError("Function currently only accepts equal width and height")

    # arrange tiles in dict according to dim for variable assignment
    tile_dict = {"arr{0}".format(i): arr[i] for i in range(dim[0]*dim[1])}
    combined, final_list = [], []

    # slice dictionary to get needed array dimensions for eventual numpy array creation
    for i in range(dim[0]):
        x = list(dict(itertools.islice(tile_dict.items(), i*dim[0], (i * dim[0]) + dim[0])).values())
        combined = combined + x

    for i in range(dim[1]):
        x = np.concatenate((combined[i*dim[0]:(i*dim[0]) + dim[0]]), axis=1)
        final_list.append(x)

    original_image = np.concatenate((final_list[:]), axis=0)

    return original_image

def infer_petri(image, petri, confidence):
    img = cv2.imread(image)
    petri_results = petri(image, conf=confidence)
    coords, count = count_objects(petri_results)
    #show_results(petri_results)
    return coords, count

def infer_phage(image, plaque):
    return plaque(image, conf=0.5)
    
def petri_crop(image, petri_count, petri_coords):
    if petri_count == 1:
        return crop_box(image, petri_coords)
    else:
        return image

def count_objects(results):
    coords = [i.xyxy.tolist()[0] for i in results[0].boxes]
    count = len(coords)
    return coords, count

def crop_box(image: str, coords: list):
    x1,y1,x2,y2 = [int(i) for i in coords[0]]
    img = cv2.imread(image)
    cropped = img[y1:y2, x1:x2]
    #circular_crop(cropped)
    #cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)
    return cropped

def circular_crop(bounding_box_image):
    width = len(bounding_box_image)
    height = width
    r = int(height/2)
    centre = int(len(bounding_box_image)**2)/2
    print(centre)
    
    # index all pixels - no real need
    # determine centre pixel index - half of pixel number
    # use radius to determine pixels in circle (hard part)
    # replace pixels outside circle with [0,0,0]
    pass

def show_results(results):
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image

def save_results(results, output: str, noimage=False):
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        if not noimage:
            im.save(f'{output}_result.jpg')  # save image
    return im

def get_img_from_results(results):
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    return im

def get_iou(bb1: list, bb2: list):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes."""

    assert bb1[0] < bb1[2] # coord validity
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0]) # interesection coords
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top) # get area of intersection
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou

def get_box_area(bb):
    """Calculate area of a bounding box"""
    return (abs(bb[2] - bb[0]) * abs(bb[3] - bb[1]))

def check_dup_bb(bb: list, arr: list):
        if any([all([(bb[0] == i[0]) , (bb[1] == i[1]), (bb[2] == i[2]), (bb[3] == i[3])]) for i in arr]):
            return True
        return False

def filter_boxinbox(arr: list):
    """Classic IOU rules may not apply so need dedicated method here"""
    output: list = []
    print(f'input_array: {len(arr)}') # error checking 1
    for i, j in enumerate(arr):
        bb1_x0, bb1_y0, bb1_x1, bb1_y1 = j[0], j[1], j[2], j[3]
        for k in arr:
            bb2_x0, bb2_y0, bb2_x1, bb2_y1 = k[0], k[1], k[2], k[3]
            if not all([(bb1_x0 > bb2_x0), (bb1_y0 > bb2_y0), (bb1_x1 < bb2_x1), (bb1_y1 < bb2_y1)]):
                if not check_dup_bb(j, output):
                    output.append(j)
    print(f'Output_array: {len(output)}') # error checking 2
    return output
                
def bound_box_overlap(arr, overlap: float):
    """IOU calculation on bounding boxes"""
    output: list = []
    for i, j in enumerate(arr):
        if all([get_iou(j, arr[k]) < overlap for k in range(i+1,len(arr)-1)]):
            output.append(j)
    return output

def remove_false_boxes(bb, wl_ratio):
    """remove false bounding boxes"""
    return [i for i in bb if min(abs((i[0] - i[2])/(i[1] - i[3])), abs((i[1] - i[3])/(i[0] - i[2]))) >= wl_ratio] 
    
def draw_boxes(image, boxes, colour=(0,0,255)):
    for bb in boxes:
        x0, y0, x1, y1 = bb[0], bb[1], bb[2], bb[3]
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(image, start_point, end_point, color=colour, thickness=2)

    return image 

def coord_modif(coords: list, dim: list, img_width: int):
    """Modifies tile coords for whole image compatibility"""
    # get position of current tile
    x_dim, y_dim = dim[0], dim[1]
    for frame_y in range(y_dim):
        for frame_x in range(x_dim):
            tile_pos = ((frame_y * y_dim) + (frame_x)) 
            current_arr = coords[tile_pos]
            for bb in current_arr:
                bb[0] = bb[0] + (frame_x * img_width)
                bb[1] = bb[1] + (frame_y * img_width)
                bb[2] = bb[2] + (frame_x * img_width)
                bb[3] = bb[3] + (frame_y * img_width)
            coords[tile_pos] = current_arr

    return coords 

def plaque_measure(petri_size: int, coords_plaques: list):
    """Measure size of phage plaques given image"""
    ref_size: float = 90 / petri_size
    plaque_sizes: list = [abs(i[2] - i[0]) * ref_size for i in coords_plaques]
    plaque_areas: list = [(((i/2) ** 2) * math.pi) for i in plaque_sizes] 
    print(plaque_sizes)
    print(f'Average plaque Size: {(sum(plaque_sizes)/len(plaque_sizes)):.2f}mm')
    print(f'Average plaque area: {(sum(plaque_areas)/len(plaque_areas)):.2f}mm2')
    

def flatten(x):
    if x == []:
        return x
    if isinstance(x[0], list):
        return flatten(x[0]) + flatten(x[1:])
    return x[:1] + flatten(x[1:])

def load_models_image(img, model1, model2):
    petri_model = YOLO(model1) 
    plaque_model = YOLO(model2)
    return img, petri_model, plaque_model

def petri_infer_processing(img, model, confidence=0.9):
    petri_coords, petri_count = infer_petri(img, model, confidence) # detect presence of 0 or more petri dishes
    cropped_image = petri_crop(img, petri_count, petri_coords) # crop image if needed
    cropped_image = cv2.resize(cropped_image, (1664,1664)) # resizing image for compatibility
    #Circular crop could be applied here as necessary
    return cropped_image

def count_boxes(coords):
    return len(coords)

def main():
    args = arguments()
    # add petri and plaque possibilities for final tool
    if args.mode == "plaque":
        img, petri_model, plaque_model = load_models_image(args.image, "petri_model.pt", "plaque_model.pt") # loading and processing image
    elif args.mode == "colony":
            img, petri_model, plaque_model = load_models_image(args.image, "petri_model.pt", "colony1.pt") # loading and processing image
    else:
        raise IOError("Invalid model mode provided to arguments. Must choose from 'colony' or 'plaque'")  
       
    cropped_image = petri_infer_processing(img, petri_model, confidence=0.9) # petri inference
    results_whole_image = infer_phage(cropped_image, plaque_model)  # inference on whole image and return coords of boxes
    whole_image_coords = results_whole_image[0].boxes.xyxy.tolist()

    if not args.fast:
        output_images, output_image_names = tile(cropped_image, tile_size=(int(args.tilesize), int(args.tilesize)), offset=(int(args.tilesize), int(args.tilesize)),
                                                 tiledim=int(args.tiledim), imageout=False) # image tiling
        output_results: list = [results[0].boxes.xyxy.tolist() for results in list(map(partial(infer_phage, plaque=plaque_model), output_images))] # Infer on individual tiles and pull coordinates of bb
        final_coords = coord_modif(output_results, [int(args.tiledim), int(args.tiledim)], int(args.tilesize))  # modify coordinates by tile with respect to original image
        final_coords += whole_image_coords
        result_image = stitch_tiles(output_images, [int(args.tiledim), int(args.tiledim)]) # tiles have been resized so need to be stitched
    else:
        final_coords = whole_image_coords
        result_image = cropped_image
    
    flatten_coords = flatten(final_coords)

    # if required act on clean coordinates
    if args.study == "population":
        results = plaque_measure(len(result_image), final_coords)
            #with open(f'{args.output}_plaquepopulation.csv', 'w') as f:
            #    f.write(results)

    partition = bound_box_overlap([flatten_coords[i:i+4] for i in range(0, len(flatten_coords), 4)], 0.01)
    partition_sort = sorted(partition, key=cmp_to_key(lambda x, y: get_box_area(x) - get_box_area(y)))
    partition_falsebox = remove_false_boxes(partition_sort, 0.20)
    final_arr = filter_boxinbox(partition_falsebox)
    combined = draw_boxes(result_image, final_arr, colour=(255,0,255))
    modename, imagename = str(args.mode), str(args.image) # gets strings for output
    print(f'Number of {modename}: {count_boxes(final_arr)}')
    cv2.imwrite(f"{args.output}.png", combined)

if __name__ == '__main__':
    main()