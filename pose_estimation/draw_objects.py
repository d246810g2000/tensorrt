import cv2

color_tab = [
    (255, 0, 0), # "nose"
    (0, 255, 0), # "left_eye"
    (0, 0, 255), # "right_eye"
    (0, 255, 0), # "left_ear"
    (0, 0, 255), # "right_ear"
    (0, 255, 0), # "left_shoulder"
    (0, 0, 255), # "right_shoulder"
    (0, 255, 0), # "left_elbow"
    (0, 0, 255), # "right_elbow"
    (0, 255, 0), # "left_wrist"
    (0, 0, 255), # "right_wrist"
    (0, 255, 0), # "left_hip"
    (0, 0, 255), # "right_hip"
    (0, 255, 0), # "left_knee"
    (0, 0, 255), # "right_knee"
    (0, 255, 0), # "left_ankle"
    (0, 0, 255), # "right_ankle"
    (255, 0, 0), # "neck"
]

class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        
    def __call__(self, image, object_counts, objects, normalized_peaks, pt_lists=None):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])

        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                color = color_tab[j]
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    xx = float(peak[1]) * width
                    x = round(xx)
                    yy = float(peak[0]) * height
                    y = round(yy)
                    cv2.circle(image, (x, y), 3, color, 2)
                    if pt_lists is not None:
                        pt_lists[i][j * 2 + 2] = xx
                        pt_lists[i][j * 2 + 3] = yy

            for k in range(K):
                color = (255, 255, 255)
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)

