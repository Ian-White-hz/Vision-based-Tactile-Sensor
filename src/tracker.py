import cv2
import numpy as np


class Tracker:
    def __init__(self,
                 adaptive,
                 cuda,
                 lower_threshold=12.0,
                 upper_threshold=18.0):
        self.cuda = cuda
        if cuda:
           #self.__tracking_inst = cv2.cuda.FarnebackOpticalFlow_create()
           self.__tracking_inst = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        else:
            self.__tracking_inst = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.__coarse_tracking_inst = cv2.DISOpticalFlow_create(
            cv2.DISOpticalFlow_PRESET_ULTRAFAST)
        self.__fine_base_frame_queue = []
        self.__coarse_base_frame_queue = []
        self.__fine_flow_queue = []
        self.__prev_fine_flow = None
        self.__prev_coarse_flow = None
        self.__adaptive = adaptive
        self.__lower_threshold = lower_threshold  # error lower than threshold will close the loop
        self.__upper_threshold = upper_threshold  # error greater than threshold will add new reference frame

    def track(self, frame):
        downScaled_size = (20, 20)
        if self.__adaptive:
            if not self.__fine_base_frame_queue:
                x = np.linspace(0,
                                downScaled_size[0] - 1,
                                downScaled_size[0],
                                dtype='float32')
                y = np.linspace(0,
                                downScaled_size[1] - 1,
                                downScaled_size[1],
                                dtype='float32')
                self.map = np.array(np.meshgrid(x, y)).transpose((1, 2, 0))
                self.__coarse_base_frame_queue.append(
                    cv2.resize(frame,
                               downScaled_size,
                               interpolation=cv2.INTER_CUBIC))
                self.__prev_coarse_flow = np.zeros(
                    (downScaled_size[1], downScaled_size[0], 2))
                if self.cuda:
                    frame = cv2.cuda_GpuMat(frame)
                    self.__fine_flow_queue.append(
                        cv2.cuda_GpuMat(frame.size(), cv2.CV_32FC2,
                                        (0.0, 0.0)))
                    self.__prev_fine_flow = cv2.cuda_GpuMat(
                        frame.size(), cv2.CV_32FC2, (0.0, 0.0))
                else:
                    self.__fine_flow_queue.append(
                        np.zeros((frame.shape[0], frame.shape[1], 2)))
                    self.__prev_fine_flow = np.zeros(
                        (frame.shape[0], frame.shape[1], 2))
                self.__fine_base_frame_queue.append(frame)
                return self.__fine_flow_queue[-1]
            else:
                downScaled_frame = cv2.resize(frame,
                                              downScaled_size,
                                              interpolation=cv2.INTER_CUBIC)
                self.__checkLoopClosure(downScaled_frame)
                self.__prev_coarse_flow = self.__coarse_tracking_inst.calc(
                    self.__coarse_base_frame_queue[-1], downScaled_frame,
                    self.__prev_coarse_flow)
                if self.cuda:
                    self.__prev_fine_flow = self.__tracking_inst.calc(
                        self.__fine_base_frame_queue[-1],
                        cv2.cuda_GpuMat(frame), self.__prev_fine_flow)
                    final_flow = cv2.cuda.add(self.__fine_flow_queue[-1],
                                              self.__prev_fine_flow)
                else:
                    self.__prev_fine_flow = self.__tracking_inst.calc(
                        self.__fine_base_frame_queue[-1], frame,
                        self.__prev_fine_flow)
                    final_flow = self.__fine_flow_queue[
                        -1] + self.__prev_fine_flow
                error = self.__getForwardBackwardError(
                    self.__coarse_base_frame_queue[-1], downScaled_frame,
                    self.__prev_coarse_flow)
                # print("Error, ", error)
                if error > self.__upper_threshold:
                    self.__fine_base_frame_queue.append(frame)
                    self.__coarse_base_frame_queue.append(downScaled_frame)
                    self.__fine_flow_queue.append(final_flow)
                    self.__prev_fine_flow = None
                    self.__prev_coarse_flow = None
                return final_flow
        else:
            if not self.__fine_base_frame_queue:
                self.__fine_base_frame_queue.append(frame)
            self.__prev_fine_flow = self.__tracking_inst.calc(
                self.__fine_base_frame_queue[-1], frame, self.__prev_fine_flow)
            return self.__prev_fine_flow

    def __getForwardBackwardError(self, I0, I1, flow):
        flow_map = self.map + flow
        z = cv2.remap(I1,
                      flow_map[:, :, 0],
                      flow_map[:, :, 1],
                      interpolation=cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_REPLICATE)
        rev = np.abs(z.astype('float32') - I0.astype('float32'))
        rev = rev.mean()
        return rev

    def __checkLoopClosure(
        self, downScaled_frame
    ):  # update (prune) the frame queue with loop closure detection
        for i in range(len(self.__coarse_base_frame_queue) - 2, -1, -1):
            f = self.__coarse_tracking_inst.calc(
                self.__coarse_base_frame_queue[i], downScaled_frame, None)
            error = self.__getForwardBackwardError(
                self.__coarse_base_frame_queue[i], downScaled_frame, f)
            if error < self.__lower_threshold:
                print("loop closed at frame ", str(i))
                self.__fine_base_frame_queue = self.__fine_base_frame_queue[:
                                                                            i +
                                                                            1]
                self.__coarse_base_frame_queue = self.__coarse_base_frame_queue[:
                                                                                i
                                                                                +
                                                                                1]
                self.__fine_flow_queue = self.__fine_flow_queue[:i + 1]
                return True
        return False

    def reset(self):
        self.__fine_base_frame_queue = []
        self.__coarse_base_frame_queue = []
        self.__fine_flow_queue = []
        self.__prev_fine_flow = None
        self.__prev_coarse_flow = None

    def getFrameQueueLength(self):
        return len(self.__coarse_base_frame_queue)