import cv2
import numpy as np
from numpy.linalg import inv
from ObjReader import OBJ

def Project(plane = (1,0,0,0,1,0,0,0,0)):
    origin = np.array(plane[6:9])
    print(origin)

class ScreenCoord(object):
    def __init__(self,x_coord = (1,0,0),y_coord = (0,1,0), origin = (0,0,0)):
        self.origin = np.array(origin)
        self.x_coord = (np.array(x_coord) - self.origin)/np.linalg.norm(np.array(x_coord) - self.origin)
        self.y_coord = (np.array(y_coord) - self.origin)/np.linalg.norm(np.array(y_coord) - self.origin)
        self.z_coord = np.cross(self.x_coord,self.y_coord)
        self.affine_matrix = inv(np.column_stack((self.x_coord,self.y_coord,self.z_coord)))
        # print(self.affine_matrix)
        self.transform_vec = self.origin
        '''
        from world coordinates to screen coordinates
        '''

    def transform(self,A,round = True):
        if round == True:
            res = np.round((np.dot(self.affine_matrix, A.T) - self.origin.reshape([3,1])).T)
            print(res.shape)
            return res
        else:
            res = (np.dot(self.affine_matrix, A.T) - self.origin.reshape([3,1])).T
            return res


if __name__ == "__main__":
    ob = OBJ("Example02.obj")
    print(np.shape(ob.vertices))
    vs = np.array(ob.vertices)
    Project()
    cood = ScreenCoord()
    A = np.array([[3,4,8],[-4,3,8],[0,0,8],[0,0,8]])
    print(cood.transform(vs))