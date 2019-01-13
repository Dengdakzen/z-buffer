import numpy as np


class OBJ(object):
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            # elif values[0] == 'mtllib':
            #     self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append(face)
        self.vertices = np.array(self.vertices)
        self.normals = np.array(self.normals)
        self.texcoords = np.array(self.texcoords)
        self.faces = np.array(self.faces,dtype = int)


    def normalization(self,edge_length = 1000, display_ratio = 0.7):
        max_x = np.max(self.vertices[:,0])
        max_y = np.max(self.vertices[:,1])
        max_z = np.max(self.vertices[:,2])
        min_x = np.min(self.vertices[:,0])
        min_y = np.min(self.vertices[:,1])
        min_z = np.min(self.vertices[:,2])
        scale = np.max([max_x - min_x,max_y - min_y,max_z - min_z])
        print(scale)
        print([max_x ,min_x,max_y ,min_y,max_z ,min_z])

        self.vertices[:,0] = (self.vertices[:,0] - min_x)/scale*display_ratio*edge_length + edge_length * (1 - display_ratio)/2 
        self.vertices[:,1] = (self.vertices[:,1] - min_y)/scale*display_ratio*edge_length + edge_length * (1 - display_ratio)/2 
        self.vertices[:,2] = (self.vertices[:,2] - min_z)/scale*display_ratio*edge_length + edge_length * (1 - display_ratio)/2        



if __name__ == "__main__":
    filename = "Example02.obj"
    A = OBJ(filename)
    print(A.faces[0:10])