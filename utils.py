import numpy as np
from cython.anchors import anchors_cython
from cython.cpu_nms import cpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms

def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    if cpu_nms is not None:
        return _nms
    else:
        return py_nms_wrapper(thresh)

def anchors_plane(feat_h, feat_w, stride, base_anchor):
    return anchors_cython(feat_h, feat_w, stride, base_anchor)

def generate_anchors_fpn(cfg = None):
    anchors = []
    for k in cfg:
        scale_factor = 32 / k
        scales = np.array([k / scale_factor, k / (scale_factor * 2)])
        scales = scales[:,np.newaxis]
        left = 0.5 * 16 * (1 - scales)
        right = 0.5 * 16 * (1 + scales) - 1
        anchor = np.hstack((left, left, right, right))
        anchors.append(anchor.astype(np.float32))

    return anchors

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class RetinaFace_Utils:
    def __init__(self):
        self.nms_threshold = 0.4
        self.vote = False
        self.nocrop = False
        self.debug = False
        self.fpn_keys = []
        self.anchor_cfg = None
        self.preprocess = False

        self._feat_stride_fpn = [32, 16, 8]
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self._feat_stride_fpn)))

        self._num_anchors = 2 #dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.use_landmarks = True

    def detect(self, img, output, threshold=0.5, im_scale=1.0):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        
        for idx, s in enumerate(self._feat_stride_fpn):
            key = 'stride%s'%s
            stride = int(s)
            
            if self.use_landmarks:
                idx = idx*3
            else:
                idx = idx*2

            scores = output[idx].cpu().detach().numpy()
            scores = scores[: , 2:, :, :]

            idx += 1
            bbox_deltas = output[idx].cpu().detach().numpy()

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            A = 2 #self._num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            proposals = self.bbox_pred(anchors, bbox_deltas)

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]
            if stride==4 and self.decay4<1.0:
                scores *= self.decay4

            proposals[:,0:4] /= im_scale

            proposals_list.append(proposals)
            scores_list.append(scores)

            if not self.vote and self.use_landmarks:
                idx+=1
                landmark_deltas = output[idx].cpu().detach().numpy()
                landmark_pred_len = landmark_deltas.shape[1]//A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
                landmarks = self.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks[:,:,0:2] /= im_scale
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0]==0:
            if self.use_landmarks:
                landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
        if not self.vote:
            keep = self.nms(pre_det)
            det = np.hstack( (pre_det, proposals[:,4:]) )
            det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
        else:
            det = np.hstack( (pre_det, proposals[:,4:]) )
            det = self.bbox_vote(det)
        return det, landmarks

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0]
        dy = box_deltas[:, 1]
        dw = box_deltas[:, 2]
        dh = box_deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes = np.zeros(box_deltas.shape)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1]>4:
            pred_boxes[:,4:] = box_deltas[:,4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
            pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
        return pred

    def bbox_vote(self, det):
        if det.shape[0] == 0:
            dets = np.array([[10, 10, 20, 20, 0.002]])
            det = np.empty(shape=[0, 5])
        while det.shape[0] > 0:
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            merge_index = np.where(o >= self.nms_threshold)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)
            if merge_index.shape[0] <= 1:
                if det.shape[0] == 0:
                    try:
                        dets = np.row_stack((dets, det_accu))
                    except:
                        dets = det_accu
                continue
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                        axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum
        dets = dets[0:750, :]
        return dets
