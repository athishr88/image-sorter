import tkinter as tk
from tkinter.filedialog import askdirectory
import numpy as np
from PIL import ImageTk, Image
import os
import glob
from multiprocessing.dummy import Pool
import shutil
import cv2
import dlib as db
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage
import copy
import scipy.spatial.distance as dist


def sorting_menu(location):
    # region sorting_menu
    # Window contains 5 tools
    # 1. View all images 2. Find duplicate images with K-means
    # 3. Find similar photos to a photo input of our choice
    # 4. Apply filter to an image of our choice 5. Face detection program
    print(location)

    # Tasks
    def view_photos():
        # Initializing window to display the images
        gtk = tk.Toplevel()

        def getFileName(image):
            # Gets the address of the image that we have selected
            global image_pressed
            print(str(image))
            image_pressed = image
            gtk.destroy()

        # Placement of all images on different buttons on the window
        row = 0
        column = 0
        for images in os.listdir(str(location)):
            if images.endswith("jpg"):
                im = Image.open(location+'/'+images)
                im.thumbnail((64, 64))
                tkimage = ImageTk.PhotoImage(im)
                handler = lambda img=images: getFileName(img)  # here modify
                imageButton = tk.Button(gtk, image=tkimage, command=handler)  # here
                imageButton.image = tkimage
                imageButton.grid(row=row, column=column)
                if column == 8:
                    row += 1
                    column = 0
                else:
                    column += 1

        # Wait for an input
        gtk.mainloop()

        return

    def find_duplicates():
        # Function classify images into k categories based on similarity among the images
        # Histogram of each images with 10 bins is produced and HAC is applied on these histogram data
        #   to classify images and so find duplicates

        # HAC Clustering
        def HAClustering(X):
            X = np.float32(X)
            m, n = X.shape
            plt.figure(1)

            num_clusters = m
            idx = np.arange(m)

            centroids = copy.deepcopy(X)
            cluster_sizes = np.ones(m)

            dists = dist.squareform(dist.pdist(centroids))
            np.fill_diagonal(dists, float('inf'))

            min_dist = np.min(dists)
            iteration = 0

            while min_dist < 40:
                iteration += 1
                min_dist = np.min(dists)
                print(min_dist)

                i = np.where(dists == min_dist)[0][0]
                j = np.where(dists == min_dist)[0][1]

                # Make sure that i < j
                if i > j:
                    t = i
                    i = j
                    j = t
                else:
                    pass

                # temp = np.array([dists[i], dists[j]])
                # dists[i] = np.min(temp, axis=0)
                # dists = np.delete(dists, i, axis=0)
                # dists = np.delete(dists, i, axis=1)
                temp = np.array([dists[i], dists[j]])
                dists[i, :] = np.mean(temp, axis=0)
                dists[:, i] = np.mean(temp, axis=0)
                dists[j, :] = float('inf')
                dists[:, j] = float('inf')

                centroids[i] = (centroids[i] + centroids[j]) / 2
                cluster_sizes[i] += cluster_sizes[j]
                cluster_sizes[j] = 0
                centroids[j] = float('inf')
                idx[idx == idx[j]] = idx[i]

                num_clusters -= 1

            # Reindexing clusters
            u = np.unique(idx)
            for i in range(len(u)):
                idx[idx == u[i]] = i

            return idx
        # K Means clustering (Not Used)

        def KMeansCLustering(X, k):
            X = np.float32(X)
            m, n = X.shape

            step = int(m/k)
            centers = []
            for a in range(k):
                centers.append(X[int(a*step)])

            centers = np.array(centers)
            # centers = np.array(random.choices(X, k=k))

            idx = np.zeros([m])

            iter = 0
            MAX_ITER = 100

            while True:
                old_idx = idx

                # Allotting clusters
                for i in range(m):
                    min_dist = float('inf')
                    for c in range(k):
                        distance = np.linalg.norm(X[i] - centers[c])
                        if distance < min_dist:
                            min_dist = distance
                            idx[i] = c

                # Updating cluster centers
                for cen in range(k):
                    new_centre_members = []
                    for i in range(m):
                        if idx[i] == cen:
                            new_centre_members.append(X[i])
                        else:
                            pass
                    new_centre_members = np.array(new_centre_members)
                    centers[cen] = np.mean(new_centre_members, axis=0)

                # Display
                # End condition
                if np.array_equal(idx, old_idx):
                    break

                # Stop early
                iter += 1
                if iter > MAX_ITER:
                    break
            return idx

        def produce_histogram(filename):
            # Converts the full size image to 64X64 image and returns the histogram of the pixel values in 10 bins
            image = Image.open(filename)
            image.thumbnail((64,64))
            img_shape = image.size
            hist = [float(distribution)/float(img_shape[0]*img_shape[1])*100 for distribution in np.histogram(np.asarray(image), bins=10)[0]]
            return hist

        def get_data(filenames):
            # Returns an array of histograms of all photos of a folder in which photos are located
            # All 12 processors are pooled together to make computation faster
            pool = Pool(12)
            mapping = pool.map(produce_histogram, filenames)
            pool.close()
            pool.join()
            return mapping

        # def find_dupes(hists):

        # find the filenames of all photos in the selected folder
        image_paths = glob.glob(location + '/*jpg')
        num_images = len(image_paths)

        # get histogram data
        hist_data = get_data(image_paths)
        print(hist_data[0])

        # get the index of each image after HA clustering
        # k_idx = np.int32(KMeansCLustering(hist_data, k_numbers))
        k_idx = np.int32(HAClustering(hist_data))

        # Make directories for each cluster
        for i in range(len(np.unique(k_idx))):
            os.makedirs(location+'/'+str(i+1))

        # Copy photos with same cluster index into same folder
        create = shutil.copy
        for i in range(num_images):
            create(image_paths[i], location + '/' + str(k_idx[i] + 1) + '/')

        print("Done")

        return

    def similar_images():


        def view_photos1():
            gtk1 = tk.Toplevel()

            def getFileName(image_filename):
                print(str(image_filename))
                gtk1.destroy()

                # Get all images in the folder
                image_paths = glob.glob(location + '/*jpg')
                images = image_all(image_paths)

                img_file_1 = location +'/'+ image_filename

                # create an instance
                image_1 = cv2.imread(img_file_1, 0)

                for i, thumb in enumerate(images):
                    thumb = np.asarray(thumb)
                    orb_index = cv2.ORB_create()
                    k1, d1 = orb_index.detectAndCompute(image_1, None)
                    k2, d2 = orb_index.detectAndCompute(thumb, None)

                    temp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                    # matching objects
                    matching_index = temp.match(d1, d2)

                    matching_regions = []

                    for r in matching_index:
                        if (r.distance < 50):
                            matching_regions.append(r)

                    if len(matching_index) == 0:
                        print("similar_index", 0)

                    else:
                        similar_index = len(matching_regions) / len(matching_index)
                        print("similar_index:", similar_index)

                        if (similar_index > 0.1):
                            path = "similar_pics/"
                            create = shutil.copy
                            create(image_paths[i], path)

            # Display all images from which one image can be selected to find similar images of
            row = 0
            column = 0
            for images in os.listdir(str(location)):
                if images.endswith("jpg"):
                    im = Image.open(location + '/' + images)
                    im.thumbnail((64, 64))
                    tkimage = ImageTk.PhotoImage(im)
                    handler = lambda img=images: getFileName(img)  # here modify
                    imageButton = tk.Button(gtk1, image=tkimage, command=handler)  # here
                    imageButton.image = tkimage
                    imageButton.grid(row=row, column=column)
                    if column == 8:
                        row += 1
                        column = 0
                    else:
                        column += 1
            gtk1.mainloop()
            return

        def read_images(filename):
            # Read Image
            img = Image.open(filename)
            return img

        def image_all(filenames):
            # returns a list of all images
            # All 12 processors pooled to read images to increase performance
            pool = Pool(12)
            mapping = pool.map(read_images, filenames)
            pool.close()
            pool.join()

            return mapping

        # Run the program
        view_photos1()
        return

    def stylize_image():
        # Provides window to apply different types of filters on an image of the users choice
        def view_photos2():
            # Display all photos so that the user can select an image
            gtk2 = tk.Toplevel()

            def getFileName(image_filename):
                # Gets the address of the file user has selected
                print(str(image_filename))
                gtk2.destroy()

                # Read image
                img_file_1 = location + '/' + image_filename
                # rgb image
                img = cv2.imread(img_file_1)

                def vignette_filter():
                    # Displays the image with a black outline mask
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # vintage filter

                    img_shape = img.shape

                    row = img_shape[0]
                    col = img_shape[1]

                    # filter

                    kernel_1 = cv2.getGaussianKernel(col, 500)
                    kernel_2 = cv2.getGaussianKernel(row, 500)

                    main_kernel = kernel_1 * kernel_2.T

                    fil = 8000 * np.sin(np.linalg.norm(main_kernel)) * (255 * main_kernel) / (
                        np.linalg.norm(main_kernel))
                    fil = fil.T

                    print(fil.shape)
                    #
                    # vintage_image = np.zeros(img.shape)
                    vintage_image = np.copy(rgb_img)
                    vintage_image = np.int32(vintage_image)
                    print(vintage_image.shape)
                    for i in range(3):
                        vintage_image[:, :, i] = vintage_image[:, :, i] * fil

                    # Normalize
                    vintage_image[vintage_image > 255] = 255
                    vintage_image[vintage_image < 0] = 0
                    vintage_image = np.uint8(vintage_image)
                    plt.figure()
                    plt.title("Vignette Filter")
                    plt.imshow(vintage_image)
                    plt.show()

                def edges():
                    # Returns edge intensity image
                    # Edge is combination of horizontal and vertical sobel filter convolved through image
                    img = cv2.imread(img_file_1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    def gradient_x(img1):
                        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
                        img1 = np.float32(img1)
                        grad_img = ndimage.convolve(img1, kernel)
                        return grad_img

                    def gradient_y(img1):
                        kernel = np.transpose(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
                        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
                        img1 = np.float32(img1)
                        grad_img = ndimage.convolve(img1, kernel)
                        return grad_img

                    def gradient_mag(gx, gy):
                        grad_img = np.hypot(gx, gy)
                        return grad_img

                    # Calculating horizontal and vertical gradient values
                    grad_x = gradient_x(img)
                    grad_y = gradient_y(img)

                    # total gradient
                    grad = gradient_mag(grad_x, grad_y)
                    plt.figure()
                    plt.title("Edges Filter")
                    plt.imshow(grad, cmap='gray')
                    plt.show()

                def gotham_filter():
                    image_name = img_file_1

                    # Reading the image
                    image = cv2.imread(image_name, 1)

                    # convert BGR TO RGB
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # converting the int type to float type
                    img = skimage.img_as_float(img)

                    # spitting the channels
                    red_pixels = img[:, :, 0]
                    green_pixels = img[:, :, 1]
                    blue_pixels = img[:, :, 2]

                    # function for sharpening the image
                    def sharpe_image(img, a, b):
                        # blur the image first
                        blur_image = cv2.GaussianBlur(img, (5, 5), 0)

                        # sharpening the image
                        sharp_image = np.clip(filtered_image * a - blur_image * b, 0, 1.0)

                        return sharp_image

                    # interpolation of red pixels

                    data_points = []
                    for i in range(11):
                        data_points.append(i * 0.1)

                    temp_red = np.interp(red_pixels.flatten(), np.linspace(0, 1, len(data_points)), data_points)
                    red_pixels = temp_red.reshape(red_pixels.shape)

                    # increase the color of blue
                    blue_pixels = np.clip(blue_pixels + 0.2, 0, 1.0)

                    # increse the color of green
                    # green_pixels = np.clip(green_pixels+0.1,0,1.0)

                    # filtered image
                    filtered_image = np.stack([red_pixels, green_pixels, blue_pixels], axis=2)

                    # sharpening the image
                    sharp_image = sharpe_image(filtered_image, 1.5, 0.5)

                    # spitting the channels
                    red_pixels = sharp_image[:, :, 0]
                    green_pixels = sharp_image[:, :, 1]
                    blue_pixels = sharp_image[:, :, 2]

                    # interpolation of blue pixels
                    data_points = [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671,
                                   0.733, 0.847, 0.925, 1]

                    temp_blue = np.interp(blue_pixels.flatten(), np.linspace(0, 1, len(data_points)), data_points)

                    blue_pixels = temp_blue.reshape(blue_pixels.shape)

                    # final filter
                    final_filter_image = np.stack([red_pixels, green_pixels, blue_pixels], axis=2)

                    # show the final filtered image
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title("Original image")
                    plt.subplot(1, 2, 2)
                    plt.imshow(final_filter_image)
                    plt.title("Gotham Filter image")
                    plt.show()

                # Pop up window of different types of filter selection
                filters_window = tk.Toplevel()
                filters_window.title("Filters")

                filter_commands = [vignette_filter, edges, gotham_filter]
                filter_buttons = ["Vignette Filter", "Edges Filter", 'Gotham Filter']

                for button3 in range(len(filter_buttons)):
                    create1 = tk.Button(master=filters_window, relief=tk.RAISED, borderwidth=2, command=filter_commands[button3], height=5, bg='#291e1d', fg='white', text=filter_buttons[button3], font=('Helvetica 18 bold', 15))
                    create1.grid(row=0, column=button3)

                filters_window.mainloop()

            # Window to show all photos. User can select a photo to apply filter
            row = 0
            column = 0
            for images in os.listdir(str(location)):
                if images.endswith("jpg"):
                    im = Image.open(location + '/' + images)
                    im.thumbnail((64, 64))
                    tkimage = ImageTk.PhotoImage(im)
                    handler1 = lambda img=images: getFileName(img)  # here modify
                    imageButton = tk.Button(gtk2, image=tkimage, command=handler1)  # here
                    imageButton.image = tkimage
                    imageButton.grid(row=row, column=column)
                    if column == 8:
                        row += 1
                        column = 0
                    else:
                        column += 1
            gtk2.mainloop()

        view_photos2()
        return

    def people_photos():
        def read_images(filename):
            img = cv2.imread(filename, 0)
            # img.thumbnail((64, 64))
            return img

        def image_all(filenames):
            pool = Pool(12)
            mapping = pool.map(read_images, filenames)
            pool.close()
            pool.join()

            return mapping

        image_paths = glob.glob(location+'/*jpg')
        all_images = image_all(image_paths)

        for i, image_one in enumerate(all_images):
            face_detect = db.get_frontal_face_detector()

            all_faces = face_detect(image_one)

            print(all_faces)

            if len(all_faces) > 0:
                path = "Faces/"
                create = shutil.copy
                create(image_paths[i], path)

        return

    # Back and close button functions
    def go_back():
        sort_window.destroy()
        main_menu()

    def close():
        sort_window.destroy()

    # Displays window to select the type of function needed by the user to apply for their photos
    sort_window = tk.Tk()
    sort_window.title("Tools")
    logo1 = ImageTk.PhotoImage(file='Logo.png')

    logo_frame = tk.Frame(master=sort_window)
    logo_frame.pack()
    logo_label1 = tk.Label(master=logo_frame, image=logo1)
    logo_label1.pack(side=tk.TOP)

    m2_back_button = tk.Button(master=logo_frame, text="Go Back", command= go_back)
    m2_back_button.pack()
    m2_close_button = tk.Button(master=logo_frame, text="Close", command=close)
    m2_close_button.pack()

    m2_command_frame = tk.Frame(master=sort_window)
    m2_command_frame.pack()

    commands = [view_photos, find_duplicates, similar_images, stylize_image, people_photos]
    buttons = ["View All Photos", "Find Duplicate Photos", "Find similar images", "Stylize Images", "See all photos with face"]
    for button2 in range(len(buttons)):
        create = tk.Button(relief=tk.RAISED, borderwidth=2, command=commands[button2], master=m2_command_frame, height=5, bg='#291e1d', fg='white', text=buttons[button2], font=('Helvetica 18 bold', 15))
        create.grid(row=0, column=button2)
    sort_window.mainloop()


# region main_menu of the GUI
def main_menu():
    # Command for open button
    def open_folder():
        mm_entry.delete(0, tk.END)
        folder_path = askdirectory()
        mm_entry.insert(0, folder_path)


    def copy_location():
        # Extract the folder location in the entry box
        photos_location = mm_entry.get()
        main_menu.destroy()
        sorting_menu(photos_location)


    def mm_destroy():
        # Command for cancel button
        main_menu.destroy()

    # Initializing the main menu window with logo
    main_menu = tk.Tk()
    logo = ImageTk.PhotoImage(file='Logo.png')

    main_menu.title("Select Directory")

    # Placement of labels buttons and entry boxes in the window
    open_frame = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
    open_frame.pack(fill=tk.X)

    logo_label = tk.Label(master=open_frame, image=logo)
    logo_label.grid(row=0, column=0)
    logo_line = tk.Label(master=open_frame, text="Image Sorter", font='Helvetica 18 bold')
    logo_line.grid(row=0, column=1, sticky='w')

    mm_label = tk.Label(master=open_frame, text="Select the directory of Photos")
    mm_label.grid(row=1, column=1, sticky='w')
    load_button = tk.Button(master=open_frame, text="Open", command=open_folder)
    load_button.grid(row=2, column=0, sticky='ew')

    mm_entry = tk.Entry(master=open_frame, width=80)
    mm_entry.grid(row=2, column=1)

    command_frame = tk.Frame( borderwidth=3)
    command_frame.pack(fill=tk.X)

    mm_ok_button = tk.Button(master=command_frame, text='OK', command=copy_location)
    mm_ok_button.pack(side=tk.RIGHT)
    mm_cancel_button = tk.Button(master=command_frame, text='Cancel', command=mm_destroy)
    mm_cancel_button.pack(side=tk.RIGHT)

    # Wait for an input
    main_menu.mainloop()

    # endregion


# Run the program
main_menu()

