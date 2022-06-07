# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import os
from matplotlib.widgets import Button, TextBox, RectangleSelector, RadioButtons
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


class HoleLocator(object):
    # This class runs a simple GUI tool to identify holes in an image 
    # and export the locations to various file formats. It runs on initialization, 
    # all it needs is the file path. 
    
    def __init__(self, image_path, dot_size=3):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.roi = None
        self.cx = None
        self.cy = None
        self.pixsize = None

        self.fig = plt.figure(figsize=(8,9))
        
        gs = GridSpec(4, 1, height_ratios=(10,1,1,1))
        self.img_ax = self.fig.add_subplot(gs[0])
        
        gs2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1])
        self.dotsize_ax = self.fig.add_subplot(gs2[0])
        self.originx_ax = self.fig.add_subplot(gs2[1])
        self.originy_ax = self.fig.add_subplot(gs2[2])
        
        gs3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2])
        self.pixsize_ax = self.fig.add_subplot(gs3[0])
        self.units_ax = self.fig.add_subplot(gs3[1])
        
        gs4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], width_ratios=(3,1))
        self.filepath_ax = self.fig.add_subplot(gs4[0])
        self.savefile_ax = self.fig.add_subplot(gs4[1])

        self.d = dot_size
        self.threshold_constant = 10

        self.img_ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        self.dotsize_textbox = TextBox(self.dotsize_ax, "Dot size \n[px]")
        self.dotsize_textbox.on_submit(self.locatedots_callback)
        self.roisel = RectangleSelector(self.img_ax, self.roi_callback,
                                        interactive=True, useblit=True,
                                        rectprops = dict(facecolor='white', 
                                                         edgecolor = 'red', 
                                                         alpha=0.1, fill=True))
        
        self.filepath_textbox = TextBox(self.filepath_ax, "Output \nfilename")

        self.pixsize_textbox = TextBox(self.pixsize_ax, "Pixel size: ")
        self.pixsize_textbox.on_submit(self.pixsize_callback)
        
        self.units_buttons = RadioButtons(self.units_ax, ('[mils]', '[mm]'))

        self.originx_textbox = TextBox(self.originx_ax, "Cx \n[px]")
        self.originx_textbox.on_submit(self.originx_callback)
        self.originy_textbox = TextBox(self.originy_ax, "Cy \n[px]")
        self.originy_textbox.on_submit(self.originy_callback)

        self.savefile_button = Button(self.savefile_ax, 'Save data')
        self.savefile_button.on_clicked(self.save_data_callback)
        
        self.fig.show()
        self.dotsize_textbox.set_val(str(self.d))
        self.originx_textbox.set_val(str(0))
        self.originy_textbox.set_val(str(0))
        self.pixsize_textbox.set_val(str(1))
        self.locatedots_callback(self.dotsize_textbox.text)
        

    def locate_holes(self):
        '''This is the workhorse function for finding the dot locations. 
           In short, the algorithm is as follows:
               1) Convert image to grayscale
               2) Perform adaptive threshold and locate contours for rough positioning
               3) Filter to size consistent with expected dot size
               4) In neighborhood around dot, threshold to minimum value to 
                  suppress noise. Subtract minimum value to make baseline zero.
               5) Calculate center of mass of remaining pixels'''
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        window = round(3*self.d)
        if window % 2 == 0:
            window += 1 # Window is required to be odd number in width

        threshInv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                             cv2.THRESH_BINARY_INV, window, self.threshold_constant) 

        cnts = cv2.findContours(threshInv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        d1 = 0.6 * self.d
        d2 = 1.4 * self.d

        s1 = np.pi * (d1/2)**2
        s2 = np.pi * (d2/2)**2
        
        xcnts = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if s1 < area < s2:
                xcnts.append(cnt)
    
        data = []
        for cnt in xcnts:
            try:
                M = cv2.moments(cnt)
                cX = (M['m10'] / M['m00'])
                cY = (M['m01'] / M['m00'])
                
                if self.roi is not None:
                    roix1, roiy1, roix2, roiy2 = self.roi
                    if not(roix1 < cX < roix2) or not(roiy1 < cY < roiy2):
                        continue
        
                subimage = gray[int(cY-self.d):int(cY+self.d+1), int(cX-self.d):int(cX+self.d+1)]
                
                X1, Y1 = np.meshgrid(np.arange(int(cY-self.d),int(cY+self.d+1)), 
                                     np.arange(int(cX-self.d),int(cX+self.d+1)))
        
                minv = subimage.min()
                maxv = subimage.max()
                dif = maxv - minv
                thres = maxv - dif/5
                subimage[subimage > thres] = maxv
                subimage = maxv - subimage
                m = cv2.moments(subimage)
                scx = m['m10'] / m['m00']
                scy = m['m01'] / m['m00']
                
                cxnew = int(cX-self.d) + scx
                cynew = int(cY-self.d) + scy
                data.append((cxnew, cynew))
            except ValueError:
                continue
    
        data = np.array(data)
        ind = np.lexsort((np.floor(data[:,1]), np.floor(data[:,0])))
        
        return data[ind]
    
    def locatedots_callback(self, expression):
        xl = self.img_ax.get_xlim()
        yl = self.img_ax.get_ylim()
        self.img_ax.clear()
        
        if self.roi is not None:
            self.rect = patches.Rectangle((self.roi[0],self.roi[1]),
                                          self.roi[2]-self.roi[0],
                                          self.roi[3]-self.roi[1], 
                                          edgecolor='r', facecolor="none")
            self.img_ax.add_patch(self.rect) 
        
        self.img_ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        self.img_ax.set_xlim(xl)
        self.img_ax.set_ylim(yl)
        
        try:
            self.d = float(expression)
            self.data = self.locate_holes()
            self.img_ax.scatter(self.data[:,0], self.data[:,1], color='red', s=3)
            self.img_ax.set_title('%d dots located'%len(self.data))
        except IndexError:
            self.data = None
            self.img_ax.set_title('No dots located')
        except:
            self.data = None
            self.img_ax.set_title('Input error, please enter a positive number')
        plt.draw()
        return
    
    def pixsize_callback(self, expression):
        try:
            self.pixsize = float(expression)
        except:
            self.img_ax.set_title('Input error, please enter a valid number')
    
    def originx_callback(self, expression):
        try:
            self.cx = float(expression)
        except:
            self.img_ax.set_title('Input error, please enter a valid number')
    
    def originy_callback(self, expression):
        try:
            self.cy = float(expression)
        except:
            self.img_ax.set_title('Input error, please enter a valid number')
    
    
    def roi_callback(self, vertex1, vertex2): # Not gonna work!!!
        x1 = vertex1.xdata
        y1 = vertex1.ydata
        x2 = vertex2.xdata
        y2 = vertex2.ydata
        self.roi = [x1, y1, x2, y2]
        xlim = self.img_ax.get_xlim()
        ylim = self.img_ax.get_ylim()
        self.img_ax.clear()
        self.img_ax.set_title('ROI changed')
        self.img_ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        rect = patches.Rectangle((self.roi[0],self.roi[1]),
                                 self.roi[2]-self.roi[0],self.roi[3]-self.roi[1],
                                 edgecolor='r', facecolor="none")
        self.img_ax.add_patch(rect) 
        self.img_ax.set_xlim(xlim)
        self.img_ax.set_ylim(ylim)
        self.locatedots_callback(str(self.d))
        return

    def save_data_callback(self, evnt):
        # Determine absolute vs relative path
        outfile_path = self.filepath_textbox.text
        if outfile_path == '':
            self.img_ax.set_title('Error saving data: enter filename')
            return
        if os.path.isabs(outfile_path):
            full_outfile_path = outfile_path
        else:
            cwd = os.getcwd()
            full_outfile_path = os.path.join(cwd, outfile_path)
        
        if self.data is None:
            self.img_ax.set_title('Error saving data: no data')
        else:
            ext = os.path.splitext(full_outfile_path)[-1].lower()
            if ext == '.txt':
                self.img_ax.set_title('Saving tab-separated data to:\n %s'%full_outfile_path)
                try:
                    cx = 0 if self.cx is None else self.cx
                    cy = 0 if self.cy is None else self.cy
                    pixsize = 1 if self.pixsize is None else self.pixsize
                    scaled_data = (self.data - np.array([cx, cy]))*pixsize
                    np.savetxt(full_outfile_path, scaled_data, delimiter='\t', 
                               header='X {0}\tY {0}'.format(self.units_buttons.value_selected))
                except PermissionError:
                    self.img_ax.set_title('Error saving data: permission denied')
            elif ext == '.csv':
                self.img_ax.set_title('Saving CSV data to:\n %s'%full_outfile_path)
                try:
                    cx = 0 if self.cx is None else self.cx
                    cy = 0 if self.cy is None else self.cy
                    pixsize = 1 if self.pixsize is None else self.pixsize
                    scaled_data = (self.data - np.array([cx, cy]))*pixsize
                    print(cx, cy, pixsize)
                    np.savetxt(full_outfile_path, scaled_data, delimiter=',',
                               header='X {0},Y {0}'.format(self.units_buttons.value_selected))
                except PermissionError:
                    self.img_ax.set_title('Error saving data: permission denied')
            else:
                self.img_ax.set_title('Error saving data: filetype must be *.csv or *.txt')
        return


filepath = './dots_original.jpg'; dot_size = 3.5
# filepath = './sim_2px.png'; dot_size = 1.5
# filepath = './sim_50px.png'; dot_size = 50

if __name__ == '__main__':
    HoleLocator(filepath, dot_size)



    
