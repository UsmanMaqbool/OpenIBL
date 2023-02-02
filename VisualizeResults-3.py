import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
from PIL import Image as PILImage
import Model as Net
import os
import time
from argparse import ArgumentParser
from torchvision.ops import masks_to_boxes
from copy import deepcopy
import torch
from torchvision.ops import batched_nms
from torchvision import transforms
from PIL import Image
import cv2
from skimage import measure
from torchvision.utils import save_image
# from cc_torch import connected_components_labeling

def heatmap_to_bboxes(heatmap, pos_thres=0.5, nms_thres=0.5, score_thres=0.5):
    """Cluster heatmap into discrete bounding boxes

    :param torch.Tensor[H, W] heatmap: Predicted probabilities
    :param float pos_thres: Threshold for assigning probability to positive class
    :param Optional[float] nms_thres: Threshold for non-max suppression (or ``None`` to skip)
    :param Optional[float] score_thres: Threshold for final bbox scores (or ``None`` to skip)
    :return Tuple[torch.Tensor]: Containing
        * bboxes[N, C=4]: bounding box coordinates in ltrb format
        * scores[N]: confidence scores (averaged across all pixels in the box)
    """

    def get_roi(data, bounds):
        """Extract region of interest from a tensor

        :param torch.Tensor[H, W] data: Original data
        :param dict bounds: With keys for left, right, top, and bottom
        :return torch.Tensor[H', W']: Subset of the original data
        """
        compound_slice = (
            slice(bounds['top'], bounds['bottom']),
            slice(bounds['left'], bounds['right']))
        return data[compound_slice]

    def is_covered(x, y, bbox):
        """Determine whether a point is covered/inside a bounding box

        :param int x: Point x-coordinate
        :param int y: Point y-coordinate
        :param torch.Tensor[int(4)] bbox: In ltrb format
        :return bool: Whether all boundaries are satisfied
        """
        left, top, right, bottom = bbox
        bounds = [
            x >= left,
            x <= right,
            y >= top,
            y <= bottom]
        return all(bounds)

    # Determine indices of each positive pixel
    heatmap_bin = torch.where(heatmap > pos_thres, 1, 0)
    mask = torch.ones(heatmap.size()).type_as(heatmap)
    idxs = torch.flip(torch.nonzero(heatmap_bin*mask), [1])
    heatmap_height, heatmap_width = heatmap.shape

    # Limit potential expansion to the heatmap boundaries
    edge_names = ['left', 'top', 'right', 'bottom']
    limits = {
        'left': 0,
        'top': 0,
        'right': heatmap_width,
        'bottom': heatmap_height}
    bboxes = []
    scores = []

    # Iterate over positive pixels
    for x, y in idxs:

        # Skip if an existing bbox already covers this point
        already_covered = False
        for bbox in bboxes:
            if is_covered(x, y, bbox):
                already_covered = True
                break
        if already_covered:
            continue

        # Start by looking 1 row/column in every direction and iteratively expand the ROI from there
        incrementers = {k: 1 for k in edge_names}
        max_bounds = {
            'left': deepcopy(x),
            'top': deepcopy(y),
            'right': deepcopy(x),
            'bottom': deepcopy(y)}
        while True:

            # Extract the new, expanded ROI around the current (x, y) point
            bounds = {
                'left': max(limits['left'], x - incrementers['left']),
                'top': max(limits['top'], y - incrementers['top']),
                'right': min(limits['right'], x + incrementers['right'] + 1),
                'bottom': min(limits['bottom'], y + incrementers['bottom'] + 1)}
            roi = get_roi(heatmap_bin, bounds)

            # Get the vectors along each edge
            edges = {
                'left': roi[:, 0],
                'top': roi[0, :],
                'right': roi[:, -1],
                'bottom': roi[-1, :]}

            # Continue if at least one new edge has more than ``pos_thres`` percent positive elements
            # Also check whether ROI has reached the heatmap boundary
            keep_going = False
            for k, v in edges.items():
                if v.sum()/v.numel() > pos_thres and limits[k] != max_bounds[k]:
                    keep_going = True
                    max_bounds[k] = bounds[k]
                    incrementers[k] += 1

            # If none of the newly expanded edges were useful
            # Then convert the maximum ROI to bbox and calculate its confidence
            # Single pixel islands are ignored since they have zero width/height
            if not keep_going:
                final_roi = get_roi(heatmap, max_bounds)
                if final_roi.numel() > 0:
                    bboxes.append([max_bounds[k] - 1 if i > 1 else max_bounds[k] 
                                   for i, k in enumerate(edge_names)])
                    scores.append(final_roi.mean())
                break

    # Type conversions and optional NMS + score filtering
    bboxes = torch.tensor(bboxes).type_as(heatmap)
    scores = torch.tensor(scores).type_as(heatmap)
    if nms_thres is not None:
        class_idxs = torch.zeros(bboxes.shape[0])
        keep_idxs = batched_nms(bboxes.float(), scores, class_idxs, iou_threshold=nms_thres)
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
    if score_thres is not None:
        high_confid = scores > score_thres
        bboxes = bboxes[high_confid]
        scores = scores[high_confid]
    return bboxes, scores


#Road, Sidewalk, building, wall, fence pole traffic light traffic signal vegetation terrain sky person rider car truck bus train motocycle bicyle unknown

pallete = [[128, 64, 128], #Road 0
           [244, 35, 232], #Sidewalk 1 
           [70, 70, 70], #building 2
           [102, 102, 156],  #3wall 3
           [190, 153, 153],  #fence #4
           [153, 153, 153],  #pole 5
           [250, 170, 30],  #traffic light 6
           [220, 220, 0],   #traffic signal7
           [107, 142, 35], #vegetation #8
           [152, 251, 152], #terrain #9
           [70, 130, 180], #sky #10
           [220, 20, 60], #person #11
           [255, 0, 0], #rider 12
           [0, 0, 142], #cars 13
           [0, 0, 70], #truck 14
           [0, 60, 100], #bus 15
           [0, 80, 100], #train 16
           [0, 0, 230], #motorcycle 17
           [119, 11, 32], #bicycle 18
           [0, 0, 0]] #unknown 19



def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/utils.py
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

# 0 1, 2 3 4, 5 6 7, 8 9,  10 

def relabel_merge(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 255
    img[img == 17] = 255
    img[img == 16] = 255
    img[img == 15] = 255
    img[img == 14] = 255
    img[img == 13] = 255
    img[img == 12] = 255
    img[img == 11] = 255
    img[img == 10] = 10
    img[img == 9] = 9
    img[img == 8] = 9
    img[img == 7] = 7
    img[img == 6] = 7
    img[img == 5] = 7
    img[img == 4] = 4
    img[img == 3] = 4
    img[img == 2] = 4
    img[img == 1] = 1
    img[img == 0] = 1
    img[img == 255] = 0
    return img

def cropimage(img, box):
    
    '''
    This function is used to crop all regions
    '''
    x,y,w,h = box
    crop_shape = [w-x,h-y]
    empty_img = np.zeros((crop_shape.shape[1], crop_shape.shape[2], crop_shape.shape[0]), dtype=np.uint8)
    
    
def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def evaluateModel(args, model, up, image_list):
    # gloabl mean and std values
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()
    
    x = []
    
    for fname in image_list:
        img = Image.open(fname)
        x.append(to_tensor(img))
    batch = torch.stack(x)
    img_out = model(batch.cuda())  #36,20,480,640
    mask = img_out.max(1)[1]   #torch.Size([36, 480, 640])
    
    # img_test = torch.zeros(480,640)
    
    # img_test = torch.unsqueeze(batch[0],0)
    # mg_out = model(img_test.cuda())  #36,20,480,640
    # mask_t = mg_out[0].max(0)[1]  #[512, 1024]
    
          
    
    
    # patch_mask = torch.zeros((480, 640))
    for jj in range(len(mask)):  #batch processing
        
        img_orig = to_tensor(Image.open(image_list[jj]).convert('RGB'))
        name = image_list[jj].split('/')[-1]

        # single_label_mask = relabel_merge(mask[jj])    # single image mask
        # all the labels to single slides
        single_label_mask = mask[jj]
        
        # obj_ids = torch.unique(single_label_mask)
        obj_ids, obj_i = single_label_mask.unique(return_counts=True)
        obj_ids = obj_ids[1:] 
        obj_i = obj_i[1:]
        
        masks = single_label_mask == obj_ids[:, None, None]
        boxes_t = masks_to_boxes(masks.to(torch.float32))
        
        rr_boxes = torch.argsort(torch.argsort(obj_i,descending=True)) # (decending order)


        # boxes = boxes_t/16
        boxes = boxes_t

        _, H, W = img_orig.shape
        patch_mask = torch.zeros((H, W)).cuda()

        
        
        bb_x = [[0, 0, round(2*W/3), round(2*H/3)],  [round(W/3),  0,  W, round(2*H/3)], [0, round(H/3), round(2*W/3), H], [round(W/3), round(H/3), W, H]]
            
        patch_mask = torch.zeros((H, W))
        # patch_mask = torch.zeros(round(2*W/3), round(2*H/3))
        
        # rsizet = transforms.Resize((427, 320)) #H W
        rsizet = transforms.Resize((H,W)) #H W
        

        img_nodes = []
        
        
        NB = 4

        
        for idx in range(len(boxes)):
            for b_idx in range(len(rr_boxes)):
                # print(idx, " ", b_idx)
                if idx == rr_boxes[b_idx] and obj_i[b_idx] > 10000 and len(img_nodes) < NB:
                        # print("found match")
                    # print(idx, " ", b_idx)
                    patch_mask = patch_mask*0
                    # label obj_ids[rr_boxes[b_idx]]
                    patch_mask[single_label_mask == obj_ids[b_idx]] = 1
                    
                    # box boxes[rr_boxes[b_idx]]
                    # patch_mask = patch_mask.unsqueeze(0)
                    patch_maskr = rsizet(patch_mask.unsqueeze(0))
                    
                    patch_maskr = patch_maskr.squeeze(0)

                    boxesd = boxes.to(torch.long)
                        
                    x_min,y_min,x_max,y_max = boxesd[b_idx]
                
                    zero_img = patch_maskr[y_min:y_max,x_min:x_max]
                
                    # imgg = img[0].permute(1, 2, 0).numpy().astype(int)
                    c_img = img_orig[:, y_min:y_max,x_min:x_max]

                    # increase dimension
                    mmask = torch.stack((zero_img,)*3, axis=0)
        
                    # Multiply arrays
                    resultant = rsizet(c_img)
                    # resultant = c_img*mmask    
                    
                    img_nodes.append(resultant.unsqueeze(0))

                    # imgg = torch.permute(resultant, (1, 2, 0)).cpu().numpy()[0]
                    # aa = img_orig.numpy()
                    # imgg = to_image(aa)
                    
                    # cv2.imwrite(args.savedir + os.sep + 'img_'+str(idx)+'_' + name.replace(args.img_extn, 'png'), aa)
                    save_image(resultant, args.savedir + os.sep + 'img_'+str(idx)+'_' + name.replace(args.img_extn, 'png'))
                    break                    
        
        # check the size
        # print("first: ", len(img_stk))
        
        if len(img_nodes) < NB:
            for i in range(len(bb_x)-len(img_nodes)):
                x_cropped = img_orig[: ,bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                img_nodes.append(x_cropped.unsqueeze(0))
            
        

                
        print("total: ", len(img_nodes))
            
        # f_boxes = boxes
        # name = image_list[jj].split('/')[-1]

        # # print(len(boxes))
        # classMap_numpy = single_label_mask.cpu().numpy()

        # if args.colored:
        #     classMap_numpy_color = np.zeros((img_orig.shape[1], img_orig.shape[2], img_orig.shape[0]), dtype=np.uint8)
        #     for idx in range(len(pallete)): #20
        #         [r, g, b] = pallete[idx]
        #         classMap_numpy_color[classMap_numpy == idx] = [b, g, r]                    
        #     cv2.imwrite(args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png'), classMap_numpy_color)
 



def main(args):
    # read all the images in the folder
    image_list = glob.glob(args.data_dir + os.sep + '*.' + args.img_extn)
    print(image_list)
    up = None
    if args.modelType == 2:
        up = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        if args.gpu:
            up = up.cuda()

    p = args.p
    q = args.q
    classes = args.classes
    if args.modelType == 2:
        modelA = Net.ESPNet_Encoder(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = args.weightsDir + os.sep + 'encoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(
            q) + '.pth'
        print(model_weight_file)

        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    elif args.modelType == 1:
        modelA = Net.ESPNet(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = args.weightsDir + os.sep + 'decoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'
        print(model_weight_file)

        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    else:
        print('Model not supported')
    # modelA = torch.nn.DataParallel(modelA)
    if args.gpu:
        modelA = modelA.cuda()

    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    evaluateModel(args, modelA, up, image_list)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./test/data", help='Data directory')
    parser.add_argument('--img_extn', default="jpg", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--modelType', type=int, default=1, help='1=ESPNet, 2=ESPNet-C')
    parser.add_argument('--savedir', default='./test/results', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--decoder', type=bool, default=True,
                        help='True if ESPNet. False for ESPNet-C')  # False for encoder
    parser.add_argument('--weightsDir', default='./pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier. Supported only 2')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier. Supported only 3, 5, 8')
    parser.add_argument('--cityFormat', default=True, type=bool, help='If you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=20, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    assert (args.modelType == 1) and args.decoder, 'Model type should be 2 for ESPNet-C and 1 for ESPNet'
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    main(args)
