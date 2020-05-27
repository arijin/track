import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class TrackObj(object):
    def __init__(self, trackIdCount, x_0, y_0, dt=0.1, Q_std=1, R_std=0.1667):
        """Args:
        dt: 0.1   time step
        R_std: 像素点有数值化左右有0.5的偏差，std=0.5/3，使密度分布大多数落在(-0.5,0.5)区间。
        P_0要给的大一点，因为初始值是瞎造了图像的中心。
        
        """
        self.track_id = trackIdCount  # identification of each track object
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path

        self.tracker = KalmanFilter(dim_x=4, dim_z=2)
        self.tracker.F = np.array([[1, dt, 0,  0],
                                                            [0,  1, 0,  0],
                                                            [0,  0, 1, dt],
                                                            [0,  0, 0,  1]])
        self.tracker.B = np.zeros((4,1))  # 输入为0， 但是因为有计算要用到，B不能为None。

        self.tracker.u = 0.
        self.tracker.H = np.array([[1, 0, 0, 0],
                                                            [0, 0, 1, 0]])

        self.tracker.R = np.eye(2) * R_std**2
        q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
        self.tracker.Q = block_diag(q, q)
        self.tracker.x = np.array([[x_0, 0, y_0, 0]]).T
        self.tracker.P = np.eye(4) * 500.
    
    def update_(self, z):
        self.tracker.update(z)
        correct_x = np.copy(self.tracker.x)
        self.trace.append([correct_x[0,0], correct_x[2,0]])

    def predict_(self):
        self.tracker.predict()
    
    def get_prediction_(self):
        x, _ = self.tracker.get_prediction()
        return [x[0,0], x[2,0]]

class Tracker(object):
    """Tracker class that updates track vectors of object tracked"""
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                              trackIdCount, image_width, image_height, fps):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: number of objects which are being tracked.
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length

        self.image_width_ = image_width
        self.image_height_ = image_height
        self.time_interval = round(1/fps,4)

        self.tracks = []
        self.trackIdCount = trackIdCount

    def TrackOnce(self, detections):
        centers = self.ReadCenters(detections)
        self.Update(centers)

    def ReadCenters(self, detections):
        centers = []
        for detection in detections:
            if detection[0].decode()!='boat' or detection[1]<0.35:
                continue
            center = [detection[2][0], detection[2][1]]
            centers.append(center)
        return centers
    
    def Update(self, detections):
        # Create tracks if no tracks vector found
        if len(self.tracks)==0:
            for i in range(len(detections)):
                track = TrackObj(self.trackIdCount, self.image_width_//2, self.image_height_//2, dt=self.time_interval)
                track.update_(detections[i])
                self.tracks.append(track)
                self.trackIdCount += 1
            return
        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    prediction = self.tracks[i].get_prediction_()
                    diff_x, diff_y = prediction[0] - detections[j][0], prediction[1] - detections[j][1]
                    distance = np.sqrt(diff_x**2 + diff_y**2)
                    cost[i][j] = distance
                except:
                    raise Exception("Invalid!", 2)
         # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        # Identify tracks with no assignment, if any
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    self.tracks[i].skipped_frames += 1
                else:
                    self.tracks[i].skipped_frames = 0
            else:
                self.tracks[i].skipped_frames += 1
        # If tracks are not detected for long time, remove them
        # if not using `reversed`, using `del` causes confusion.
        for i in reversed(range(len(self.tracks))):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del self.tracks[i]
                del assignment[i]
        # Now look for un_assigned detects and start new tracks
        for i in range(len(detections)):
                if i not in assignment:
                    track = TrackObj(self.trackIdCount, self.image_width_//2, self.image_height_ //2)
                    track.update_(detections[i])
                    self.tracks.append(track)
                    self.trackIdCount += 1
        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].predict_()
            if assignment[i]!=-1:
                self.tracks[i].update_(detections[assignment[i]])
                if len(self.tracks[i].trace) > self.max_trace_length:
                    del self.tracks[i].trace[0]
        return    
