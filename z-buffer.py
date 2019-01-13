import os
import numpy as np
import cv2
from ObjReader import OBJ
from Render import ScreenCoord

def get_quad_rep(ThDVertices,f_set):
    v1_set = ThDVertices[f_set[:,0] - 1] - ThDVertices[f_set[:,1] - 1]
    v2_set = ThDVertices[f_set[:,0] - 1] - ThDVertices[f_set[:,1] - 2]
    v0_set = ThDVertices[f_set[:,0] - 1]
    norm_set = np.cross(v1_set,v2_set)
    d_set = -np.sum(norm_set * v0_set,axis = 1)
    # print(d_set)
    return np.column_stack((norm_set,d_set))

def Polygon_Table(TDVertices,ThDVertices, f_set):
    '''
    ymax,a,b,c,d,id,dy
    '''
    quad_rep = get_quad_rep(ThDVertices,f_set)
    ids = np.array(range(f_set.shape[0]))
    y_axis = TDVertices[f_set - 1,1]
    dy = np.max(y_axis,axis=1) - np.min(y_axis,axis=1)
    P_Set = np.column_stack((np.max(y_axis,axis=1),quad_rep,ids,dy))
    index = np.argsort(P_Set[:,0])[::-1]
    sorted_P_set = P_Set[index,:]
    return sorted_P_set 

def TwoPointsToEdge(A0,B0):
    X_max_Y_max = B0.copy()
    AB = (A0[:,1] >= B0[:,1])
    X_max_Y_max[AB] = A0[AB]
    dX_dY = A0 - B0
    sub_inv_k = -dX_dY[:,0]/dX_dY[:,1]
    dY = np.abs(dX_dY[:,1])
    ids = np.array(range(A0.shape[0]))
    return(np.column_stack((X_max_Y_max[:,1],X_max_Y_max[:,0],sub_inv_k,dY,ids)))



def TD2EdgeTable(TDVertices,f_set):
    A = TDVertices[f_set[:,0] - 1,0:2]
    B = TDVertices[f_set[:,1] - 1,0:2]
    C = TDVertices[f_set[:,2] - 1,0:2]
    D = TwoPointsToEdge(A,B)
    E = TwoPointsToEdge(C,B)
    F = TwoPointsToEdge(A,C)
    edge_set = np.row_stack((D,E,F))
    index = np.argsort(edge_set[:,0],)[::-1]
    sorted_edge_set = edge_set[index,:]
    return sorted_edge_set
    

    


class z_buffer(object):
    def __init__(self,obj,x_coord = (1,0,0),y_coord = (0,1,0), origin = (0,0,0)):
        self.data = obj
        self.plane = ScreenCoord(x_coord,y_coord,origin)
        self.f_set = np.array(self.data.faces,dtype = int)
        self.ThDVertices = np.array(self.data.vertices)

    def process(self,visualization = False):
        self.TDVertices = self.plane.transform(self.ThDVertices)
        self.f_buf = np.zeros([1000,1000],dtype = np.uint8)
        self.z_buf = np.zeros([1,1000],dtype = np.uint8)
        edge_set = TD2EdgeTable(self.TDVertices,self.f_set)
        poly_set = Polygon_Table(self.TDVertices,self.ThDVertices,self.f_set)
        print(poly_set)
        APT = []
        APT_index = []
        APT_map = np.zeros(self.f_set.shape[0])
        APT_value = poly_set[:,-1].copy()
        AET = []
        AET_index = []
        AET_value = edge_set[:,-2].copy()
        if visualization == True:
            cv2.imshow("test",self.f_buf)
            cv2.waitKey()
        current_line_y = edge_set[0][0]
        i = 0
        # for i in range(poly_set.shape[0]):
        while(1):
            if poly_set[i][0] == current_line_y:
                APT.append(poly_set[i])
                APT_index.append(i)
                # APT_map[i] = 1
                i += 1
                if i == poly_set.shape[0]:
                    break
                continue
            for poly in APT_index:
                for j in range(3):
                    if edge_set[j * poly_set.shape[0] + poly][0] >= current_line_y:


            if i == poly_set.shape[0]:
                break
            
            

        


if __name__ == "__main__":
    # a = range(10)
    # b = np.repeat(a,3)
    # print(b)
    # c = np.array([[1,1,1],[1,2,4]])
    # d = np.array([[1,7,1],[0,2,9]])
    # e = np.cross(c[0],d[0])
    # print(e)
    filename = "Example03.obj"
    A = OBJ(filename)
    # # A.normalization()
    # # plane = ScreenCoord()
    # # ThDVertices = np.array(A.vertices)
    # # TwoDvertices = plane.transform(ThDVertices)
    # # f_set = np.array(A.faces,dtype = int)
    # # # Polygon_Table(TwoDvertices,ThDVertices,f_set)
    # # K = TD2EdgeTable(TwoDvertices,f_set)
    # # print(K)
    zb = z_buffer(A)
    zb.process()
    # print(b)
    