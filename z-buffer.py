import os
import numpy as np
import cv2
from ObjReader import OBJ
from Render import ScreenCoord
import time

def get_quad_rep(ThDVertices,f_set):
    v1_set = ThDVertices[f_set[:,0] - 1] - ThDVertices[f_set[:,1] - 1]
    v2_set = ThDVertices[f_set[:,0] - 1] - ThDVertices[f_set[:,2] - 1]
    v0_set = ThDVertices[f_set[:,0] - 1]
    norm_set = np.cross(v1_set,v2_set)
    # norm_set = norm_set/(np.max(np.abs(norm_set),axis = 1)[:,np.newaxis])
    d_set = -np.sum(norm_set * v0_set,axis = 1)
    # print(d_set)
    return np.column_stack((norm_set,d_set))

def Polygon_Table(ThDVertices, f_set):
    '''
    ymax,a,b,c,d,id,dy
    '''
    # print(TDVertices)
    # print(ThDVertices)
    quad_rep = get_quad_rep(ThDVertices,f_set)
    ids = np.array(range(f_set.shape[0]))
    y_axis = ThDVertices[f_set - 1,1]
    dy = np.max(y_axis,axis=1) - np.min(y_axis,axis=1)
    P_Set = np.column_stack((np.max(y_axis,axis=1),quad_rep,ids,dy))
    index = np.argsort(P_Set[:,0])[::-1]
    sorted_P_set = P_Set[index,:]
    f_set = f_set[index,:]
    # print(f_set)
    return sorted_P_set,f_set

def TwoPointsToEdge(A0,B0):
    '''
    ymax, x_max, -1/k, dy, id
    '''
    X_max_Y_max = B0.copy()
    AB = (A0[:,1] >= B0[:,1])
    X_max_Y_max[AB] = A0[AB]
    dX_dY = A0 - B0
    # sub_inv_k = -dX_dY[:,0]/dX_dY[:,1]
    sub_inv_k = -np.divide(dX_dY[:,0], dX_dY[:,1], out=np.ones_like(dX_dY[:,0]) * 10000, where=dX_dY[:,1]!=0)
    dY = np.abs(dX_dY[:,1])
    ids = np.array(range(A0.shape[0]))
    res = np.column_stack((X_max_Y_max[:,1],X_max_Y_max[:,0],sub_inv_k,dY,ids))
    return res



def EdgeTable(ThDVertices,f_set):
    A = ThDVertices[f_set[:,0] - 1,0:2]
    B = ThDVertices[f_set[:,1] - 1,0:2]
    C = ThDVertices[f_set[:,2] - 1,0:2]
    D = TwoPointsToEdge(A,B)
    E = TwoPointsToEdge(C,B)
    F = TwoPointsToEdge(A,C)
    edge_set = np.row_stack((D,E,F))
    # index = np.argsort(edge_set[:,0],)[::-1]
    # sorted_edge_set = edge_set[index,:]
    return edge_set
    
def AET_Element(E1,E2,current_line_y,polygon,l_edge,r_edge,last_edge):
    # print(E1)
    # print(E2)
    xl = np.round((E1[0] - current_line_y)*E1[2] + E1[1])
    dxl = E1[2]
    dyl = E1[-2] - E1[0] + current_line_y
    xr = np.round((E2[0] - current_line_y)*E2[2] + E2[1])
    dxr = E2[2]
    dyr = E2[-2] - E2[0] + current_line_y
    # print(p_id)
    p_id = int(E1[-1])
    if np.abs(polygon[p_id][3]) < 0.0001:
        zl = -polygon[p_id][4]
        zr = -polygon[p_id][4]
        dzx = 0
        dzy = 0
    else:
        zl = (-polygon[p_id][4] - xl * polygon[p_id][1] - current_line_y * polygon[p_id][2])/polygon[p_id][3]
        zr = (-polygon[p_id][4] - xr * polygon[p_id][1] - current_line_y * polygon[p_id][2])/polygon[p_id][3]
        dzx = -polygon[p_id][1]/polygon[p_id][3]
        dzy = polygon[p_id][2]/polygon[p_id][3]
    if xl == xr:
        if dxl < dxr:
            return np.array([xl,dxl,dyl,xr,dxr,dyr,zl,dzx,dzy,p_id,l_edge,r_edge,last_edge])
        else:
            return np.array([xr,dxr,dyr,xl,dxl,dyl,zr,dzx,dzy,p_id,r_edge,l_edge,last_edge])
    elif xl < xr:
        return np.array([xl,dxl,dyl,xr,dxr,dyr,zl,dzx,dzy,p_id,l_edge,r_edge,last_edge])
    else:
        return np.array([xr,dxr,dyr,xl,dxl,dyl,zr,dzx,dzy,p_id,r_edge,l_edge,last_edge])

    


class z_buffer(object):
    def __init__(self,obj,x_coord = (1,0,0),y_coord = (0,1,0), origin = (0,0,0)):
        
        self.t1 = time.time()
        self.data = obj
        self.plane = ScreenCoord(x_coord,y_coord,origin)
        self.f_set = np.array(self.data.faces,dtype = int)
        self.Original_Vertices = np.array(self.data.vertices)

    def normalization(self,vertices,edge_length = 1000, display_ratio = 0.7):
            max_x = np.max(vertices[:,0])
            max_y = np.max(vertices[:,1])
            max_z = np.max(vertices[:,2])
            min_x = np.min(vertices[:,0])
            min_y = np.min(vertices[:,1])
            min_z = np.min(vertices[:,2])
            scale = np.max([max_x - min_x,max_y - min_y])
            # print(scale)
            # print([max_x ,min_x,max_y ,min_y])

            vertices[:,0] = (vertices[:,0] - 0.5*(min_x + max_x))/scale*display_ratio*edge_length + edge_length * 0.5 
            vertices[:,1] = (vertices[:,1] - 0.5*(min_y + max_y))/scale*display_ratio*edge_length + edge_length * 0.5
            vertices[:,2] = (vertices[:,2] - min_z)/(max_z - min_z)*255
            return np.round(vertices)

    def process(self,visualization = False):
        ThDVertices = self.plane.transform(self.Original_Vertices,round = False)
        ThDVertices = self.normalization(ThDVertices)
        f_buf = np.zeros([1000,1000],dtype = np.uint8)
        z_buf = np.zeros([1,1000],dtype = np.uint8)
        f_set_copy = self.f_set.copy()
        # print(f_set_copy)
        poly_set,f_set_copy = Polygon_Table(ThDVertices,f_set_copy)
        edge_set = EdgeTable(ThDVertices,f_set_copy)
        # print("edge_table:\n",edge_set)
        # print("polygon_table:\n",poly_set)
        # print(f_set_copy)
        APT = []
        APT_index = []
        APT_value = poly_set[:,-1].copy()
        AET = []
        AET_value = edge_set[:,-2].copy()
        if visualization == True:
            cv2.imshow("test",f_buf)
            cv2.waitKey()
        current_line_y = poly_set[0][0]
        least_line_y = np.min(ThDVertices[:,1])
        i = 0
        APT_to_be_delete = []
        AET_to_be_delete = []
        while(1):
            # print(i)
            if i < poly_set.shape[0] and poly_set[i][0] == current_line_y:
                APT_index.append(i)
                i += 1
                if i != poly_set.shape[0]:
                    continue
            for del_ind in APT_to_be_delete[::-1]:
                # del(APT[del_ind])
                del(APT_index[del_ind])
            APT_to_be_delete = []
            count_apt = 0
            #given activated polygon set
            for poly in APT_index:
                if poly_set[poly][-1] == APT_value[poly]:
                    two_edge = []
                    original_edge_index = [poly,poly_set.shape[0] + poly,2 * poly_set.shape[0] + poly]
                    for j in range(3):
                        if edge_set[j * poly_set.shape[0] + poly][0] >= current_line_y and np.abs(edge_set[j * poly_set.shape[0] + poly][2]) <= 1000:
                            two_edge.append(j * poly_set.shape[0] + poly)
                            original_edge_index.remove(j * poly_set.shape[0] + poly)
                    if len(two_edge) == 2:
                        last_edge = original_edge_index[0]
                        AET.append(AET_Element(edge_set[two_edge[0]],edge_set[two_edge[1]],current_line_y,poly_set,two_edge[0],two_edge[1],last_edge))
                    # print(AET)
                APT_value[poly] -= 1
                if APT_value[poly] < 0:
                    APT_to_be_delete.append(count_apt)
                count_apt += 1
            # break
            # print(AET_to_be_delete)
            for del_ind in AET_to_be_delete[::-1]:
                del(AET[del_ind])
                
            AET_to_be_delete = []
            count_aet = 0
            z_buf = np.zeros([1,1000],dtype = np.uint8)
            for et in AET:
                begin_idx = int(et[0])
                end_idx = int(et[3])
                zx = et[6]
                dzx = et[7]
                for pixel in range(begin_idx,end_idx + 1):
                    if z_buf[0,pixel] < zx:
                        z_buf[0,pixel] = zx
                    zx += dzx
                et[0] += et[1]
                et[3] += et[4]
                et[6] += et[7]*et[1] + et[8]
                et[2] -= 1
                et[5] -= 1
                if et[2] < 0 and et[5] < 0:
                    AET_to_be_delete.append(count_aet)
                elif et[2] < 0 and et[5] >= 0:
                    AET[count_aet] = AET_Element(edge_set[int(et[-2])],edge_set[int(et[-1])],current_line_y - 1,poly_set,-1,-1,-1)
                elif et[5] < 0 and et[2] >= 0:
                    AET[count_aet] = AET_Element(edge_set[int(et[-3])],edge_set[int(et[-1])],current_line_y - 1,poly_set,-1,-1,-1)

                count_aet += 1
            f_buf[int(current_line_y),:] = z_buf.copy()
            current_line_y -= 1
            if current_line_y < least_line_y:
                break
            # if i == poly_set.shape[0]:
            #     break
        f_buf = f_buf[::-1]
        self.t2 = time.time()
        print("time:",self.t2 - self.t1)
        cv2.imshow("test",f_buf)
        cv2.waitKey()
        
            
            

        


if __name__ == "__main__":

    # filename = "Example01.obj"
    filename = "bunny.obj"
    A = OBJ(filename)
    zb = z_buffer(A)
    #uncomment to change the screen coordinates
    # zb = z_buffer(A,(3,4,0),(-4,3,0))
    # zb = z_buffer(A,(1,1,1),(1,0,-1))
    zb.process()
    