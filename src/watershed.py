import cv2
import queue 
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import imageio

class Watershed:

      mask = -2 #Initial value of a threshold level
      wshed = 0 #Value of pixels belonging to watershed
      init = -1 #Initial value of f0
      inqueue = -3 #Value assigned to pixels put into the queue

      def __init__(self,image,neighNum):
            self.image = image
            self.matrix = self.getMat()
            self.current_label = 0
            self.flag = False
            self.que = queue.Queue()
            self.neighNum = neighNum

      def getMat(self):
            """
            The method returns a matrix filled with -1 values
            """
            sizeTuple = np.shape(self.image)
            self.h = sizeTuple[0]
            self.w = sizeTuple[1]
            self.size = self.h * self.w 
            mat = [[self.init for x in range(self.w)] for y in range(self.h)] 
            return mat
      
      def getNei(self, p, neighNum):
            """
            The method returns neighbour coordinated of pixels and the p, itself
            :param word: P a pixel for which neoghbour pixels are found
            :param word: NeighNum a number of connected neighborhood
            """
            if(neighNum == '8'):
                  return np.mgrid[ max(0, p[0] - 1):min(self.h, p[0]+2),
                        max(0, p[1] - 1):min(self.w, p[1]+2)].reshape(2, -1).T 
            elif(neighNum == '4'):
                  ax,ay = np.mgrid[ max(0, p[0] - 1):min(self.h, p[0]+2), p[1]:p[1]+1]
                  bx,by = np.mgrid[p[0]:p[0]+1, max(0,p[1]-1):p[1]]
                  cx,cy = np.mgrid[p[0]:p[0]+1, p[1]+1:min(self.w,p[1]+2)]

                  a_stack = np.vstack((ax.ravel(), ay.ravel()))
                  b_stack = np.vstack((bx.ravel(), by.ravel()))
                  c_stack = np.vstack((cx.ravel(), cy.ravel()))

                  return np.hstack((a_stack,b_stack,c_stack)).T
            else:
                  print("Number of connected neighborhood wasn't indicated correctly!")
                  exit

      def getIndexArray(self,array):
            """
            The method returns array of sorted elements' indices
            :param word: Array that needs to be sorted
            """
            return sorted(range(len(array)), key=array.__getitem__)

      def getUniqueValueIndex(self, array):
            """
            The method returns array of unique elements of sorted array
            :param word: Array that needs to be sorted
            """
            return np.unique(array, return_index=True)

      def flood(self):
            """
            The method returns 'watersheded' matrix 
            """
            #Pair(y,x) pixel coordinates 
            pixels = np.mgrid[0:self.h, 0:self.w].reshape(2, -1).T
            #Neighbour pixel coordinates for each pixel
            neigh = np.array([self.getNei(p,self.neighNum) for p in pixels])
            #Reshaping npArray of neighbour pixel coordinates 
            if len(neigh.shape) == 3: neigh = neigh.reshape(self.h, self.w, -1, 2)
            else: neigh = neigh.reshape(self.h, self.w)
            
            #Ordering pixel neighbors (OPTIONAL)
            h_neigh, w_neigh = np.shape(neigh)
            for i in range(0,h_neigh):
                  for j in range(0,w_neigh):
                        arr = neigh[i][j]
                        ind = np.lexsort((arr[:,1],arr[:,0]))
                        neigh[i][j] = arr[ind]

            # 1D array of the image values
            array_image = np.reshape(self.image,(self.size))
            #Indices of the sorted 1D array values
            indices = self.getIndexArray(array_image)
            #Sorted 1D array 
            sorted_array = array_image[indices]
            #Sorted pixel coordinates according to indices of the sorted 1D array 
            sorted_pixArray = pixels[indices]
            #Arrays of unique values of the sorted 1D array and their indices
            uniqVals,uniqIndices = self.getUniqueValueIndex(sorted_array)
            uniqIndices = np.delete(uniqIndices,0)
            uniqIndices = np.append(uniqIndices,self.size)
            
            first_ind = 0
            count = 0
            last_ind = uniqIndices[count]
            
            # Looping through the array of unique values 
            for h in uniqVals:
                 #Assiging masks to the pixels at the current level
                  for p in sorted_pixArray[first_ind:last_ind]:
                        self.matrix[p[0]][p[1]] = self.mask
                        for p1 in neigh[p[0],p[1]]:
                              if self.matrix[p1[0]][p1[1]] >= self.wshed:
                                    self.matrix[p[0]][p[1]] = self.inqueue
                                    self.que.put(p)
                 
                  #Assigning pixels to the current catchment basin
                  while not self.que.empty():
                        p = self.que.get()
                        for p1 in neigh[p[0],p[1]]:
                              if self.matrix[p1[0]][p1[1]] > 0:
                                    if (self.matrix[p[0]][p[1]] == self.inqueue or\
                                          (self.matrix[p[0]][p[1]] == self.wshed and self.flag == True)):
                                          self.matrix[p[0]][p[1]] = self.matrix[p1[0]][p1[1]]
                                    elif (self.matrix[p[0]][p[1]] > 0 and\
                                          self.matrix[p[0]][p[1]] != self.matrix[p1[0]][p1[1]]):
                                          self.matrix[p[0]][p[1]] = self.wshed
                                          self.flag = False
                              elif self.matrix[p1[0]][p1[1]] == self.wshed:
                                    if self.matrix[p[0]][p[1]] == self.inqueue:
                                          self.matrix[p[0]][p[1]] = self.wshed
                                          self.flag = True
                              elif self.matrix[p1[0]][p1[1]] == self.mask:
                                    self.matrix[p1[0]][p1[1]] = self.inqueue
                                    self.que.put(p1)
                  
                  #Detecting a new minima at the current level
                  for p in sorted_pixArray[first_ind:last_ind]:
                        if self.matrix[p[0]][p[1]] == self.mask:
                              self.current_label = self.current_label + 1
                              self.que.put(p)
                              self.matrix[p[0]][p[1]] = self.current_label
                              while not self.que.empty():
                                    p1 = self.que.get()
                                    for p2 in neigh[p1[0],p1[1]]:
                                          if self.matrix[p2[0]][p2[1]] == self.mask:
                                                self.que.put(p2)
                                                self.matrix[p2[0]][p2[1]] = self.current_label
                  first_ind = last_ind
                  count = count + 1
                  if(count != len(uniqIndices)):
                        last_ind = uniqIndices[count]
      
            return self.matrix

def morphTransform(image,label,name):
      """
      The function performs indicated morphological transformations
      on the input and returns the matrix
      :param word: Image a matrix of input image
      :param word: Label that corresponds to a specific morphological operation 
      """
      trans = image
      if label == 0:
            trans = cv2.medianBlur(image,5)
      else:
            kernel = np.ones((9, 9), np.uint8)
            trans = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
      
      name = name.split('.')
      imageio.imwrite('../output/'+ name[0]+'_out.png',trans)
      return trans

def distTransform(image,name):
      """
      The function returns distance transformaed matrix 
      :param word: Image a matrix of the input image
      """
      dist = ndimage.distance_transform_edt(image)
      name = name.split('.')
      imageio.imwrite('../output/'+ name[0]+'_out.png',dist)
      return dist

def invDist(image,name):
      """
      The function returns invers of the matrix 
      :param word: Image a matrix of the input image
      """
      dist = distTransform(image,name)
      inverse = (dist.max() - dist)
      inverse = np.round(inverse)
      name = name.split('.')
      imageio.imwrite('../output/'+ name[0]+'_out.png',inverse)
      return inverse

def applyWatershed(image,neighNum):
      """
      The function returns watershed transformed matrix
      :param word: Image a matrix of the input image
      :param word: NeighNum a number of connected neighborhood
      """
      w = Watershed(image, neighNum)
      return w.flood()

def transform(name,image):
      """
      The function returns transformed matrix
      :param word: Name a name of the input image
      :param word: Image a matrix of the input image
      """
      if '_dist' in name:
            return distTransform(image,name)
      elif '_dinv' in name:
            return invDist(image,name)
      elif '_median' in name:
            return morphTransform(image,0,name)
      elif '_grad' in name:
            return morphTransform(image,1,name)
      else:
            return image

userInput = input()
words = userInput.split()
numWords = len(words)

if '.txt' in words[0]:
      file = '../input/' + words[0]
      with open(file, 'r')as f:
            csv_file = csv.reader(f,delimiter=',')
            data_as_list = list(csv_file)

            image = np.array(data_as_list,dtype = np.float32)
            img = transform(words[0],image)
            final = applyWatershed(img, words[1])
            final = np.array(final,dtype = np.float32)

            output = words[0].split('.')
            with open('../output/'+ output[0]+'_wt_'+words[1]+'.txt',"w+") as my_csv:
                  csvWriter = csv.writer(my_csv,delimiter=',') 
                  csvWriter.writerows(final)
            img = cv2.convertScaleAbs(final, alpha=(255.0/final.max()))
            cv2.imwrite('../output/' + output[0] + '_wt_' + words[1] + '.png',img)
else:
      image = cv2.imread('../input/'+words[0],0)    
      image = np.array(image,dtype = np.float32) 
      img = transform(words[0],image)      
      final = applyWatershed(img, words[1])
      final = np.array(final)

      output = words[0].split('.')
      img = cv2.convertScaleAbs(final, alpha=(255.0/final.max()))
      cv2.imwrite('../output/' + output[0]+ '_wt_'+ words[1] + '.png',img)
