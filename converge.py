'''
Used for face result convergence
'''
import copy


def converge_resuts(images: dict, big_face_images: dict):
    iou_threshold = 1e-7
    results = {}
    for path in images.keys():
        results[path] = copy.deepcopy(big_face_images[path])

        for big_face_result in big_face_images[path].bbox_list:
            for images_result in images[path].bbox_list:
                if images_result.label == 'face':
                    iou = get_iou(big_face_result, images_result)
                    if iou < iou_threshold:  # OK to save
                        results[path].bbox_list.append(images_result)
                else:
                    results[path].bbox_list.append(images_result)

    return results


def get_iou(bbox1: list, bbox2: list):
    x_left = max(bbox1.x1, bbox2.x1)
    x_right = min(bbox1.x2, bbox2.x2)
    y_top = max(bbox1.y1, bbox2.y1)
    y_bottom = min(bbox1.y2, bbox2.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
    bb2_area = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou
