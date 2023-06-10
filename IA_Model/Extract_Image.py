import cv2 as cv
import  numpy as np
import hashlib as hb
import secrets as sct
import matplotlib.pyplot as plt
def extract(image_path):
    image =cv.imread(image_path)
    cv.imshow('Image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def detect_img(image_path):
    img= cv.imread(image_path)
    #image= cv.GaussianBlur(src=image_path, ksize=(5,5) ,sigmaX=1.0)
    gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    edges=cv.Canny(gray,100,200)
    cv.imshow('Original',img)
    cv.imshow('Edges',edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detect_contour(image_path):
    image =cv.imread(image_path)
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    _,binary= cv.threshold(gray,127,255,cv.THRESH_BINARY)

    kernel=np.ones((5,5), np.uint8)
    dilated=cv.dilate(binary,kernel,iterations=1)
    contour , _ =cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    image_with_contour=image.copy()
    cv.drawContours(image_with_contour,contour,-1,(0,255,0),2)

    cv.imshow('Original',image)
    cv.imshow('Image_eith_contour',image_with_contour)
    cv.waitKey(0)
    cv.destroyAllWindows()
def bound_boxing(image_path):
    image =cv.imread(image_path)

    if image is not None :
        boundng_box=np.array([[0,0],])
def find_mathing_image(image_path,targer_path):
    image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    target=cv.imread(targer_path,cv.IMREAD_GRAYSCALE)
    # create sift instance
    sift =cv.SIFT()
    keypoint_image,description_image=sift.detectAndCompute(image,None)
    keypoint_target,description_target=sift.detectAndCompute(target,None)
    matcher= cv.BFMatcher()
    matches = matcher.match(description_image,description_target)
    matches=sorted(matches, key=lambda x:x.distance)
    matched_image = cv.drawMatches(image, keypoint_image, target, keypoint_target, matches[:10], None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matched Image", matched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def find_matching_image2(template_path, target_path):
    # Read the template and target images
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    target = cv.imread(target_path, cv.IMREAD_GRAYSCALE)

    # Create an ORB object
    orb = cv.ORB_create()

    # Detect keypoints and compute descriptors for the template and target images
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
    keypoints_target, descriptors_target = orb.detectAndCompute(target, None)

    # Create a Brute-Force Matcher object
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match the descriptors of the template and target images
    matches = matcher.match(descriptors_template, descriptors_target)

    # Sort the matches by their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top matching keypoints on the target image
    matched_image = cv.drawMatches(template, keypoints_template, target, keypoints_target, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image
    cv.imshow("Matched Image", matched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()





def generate_artwork_token(artwork_data):
    # Convert artwork data to a unique string representation
    artwork_string = str(artwork_data)

    # Generate a unique hash for the artwork data
    hash_object = hb.sha256(artwork_string.encode())
    hash_digest = hash_object.digest()

    # Generate a random token
    token = sct.token_hex(16)

    # Combine the hash and token to form the final artwork token
    artwork_token = hash_digest + token.encode()
    print('this is token '+str(artwork_token))
    return artwork_token





def extract_image_from_background(image_path):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image from BGR to RGB color space
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the green color range
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)

    # Create a mask for the green pixels in the image
    mask = cv.inRange(image, lower_green, upper_green)

    # Apply the mask to extract the foreground image
    foreground = cv.bitwise_and(image, image, mask=mask)
    cv.imshow('Background', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return foreground




def extract_image_from_background2(image_path):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image from BGR to RGB color space
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the green color range
    lower_green = (0, 100, 0)
    upper_green = (100, 255, 100)

    # Create a mask for the green pixels in the image
    mask = cv.inRange(image_rgb, lower_green, upper_green)

    # Apply the mask to extract the foreground image
    foreground = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Create a subplot to display the original image and the extracted foreground
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display the extracted foreground image
    ax[1].imshow(foreground)
    ax[1].set_title('Extracted Foreground')
    ax[1].axis('off')

    # Show the plot
    plt.show()





def extract_and_contour_image3(image_path):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image from BGR to RGB color space
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the green color range
    lower_green = (0, 100, 0)
    upper_green = (100, 255, 100)

    # Create a mask for the green pixels in the image
    mask = cv.inRange(image_rgb, lower_green, upper_green)

    # Apply the mask to extract the foreground image
    foreground = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Convert the extracted foreground to grayscale
    gray = cv.cvtColor(foreground, cv.COLOR_RGB2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    image_contoured = cv.drawContours(image_rgb, contours, -1, (255, 0, 0), 2)

    # Create a subplot to display the original image, the extracted foreground, and the contoured image
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Display the original image
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display the extracted foreground image
    ax[1].imshow(foreground)
    ax[1].set_title('Extracted Foreground')
    ax[1].axis('off')

    # Display the contoured image
    ax[2].imshow(image_contoured)
    ax[2].set_title('Contoured Image')
    ax[2].axis('off')

    # Show the plot
    plt.show()





def extract_and_contour_main_object(image_path):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image from BGR to RGB color space
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)


    # Define the lower and upper bounds for the green color range
    lower_green = (0, 100, 0)
    upper_green = (100, 255, 100)

    # Create a mask for the green pixels in the image
    mask = cv.inRange(image_rgb, lower_green, upper_green)

    # Apply the mask to extract the foreground image
    foreground = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Convert the extracted foreground to grayscale
    gray = cv.cvtColor(foreground, cv.COLOR_RGB2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (main object)
    main_contour = max(contours, key=cv.contourArea)

    # Create a copy of the original image
    image_with_contour = image_rgb.copy()

    # Draw the contour of the main object on the image
    cv.drawContours(image_with_contour, [main_contour], -1, (255, 0, 0), 2)

    # Create a subplot to display the original image, the extracted foreground, and the image with the contour
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Display the original image
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display the extracted foreground image
    ax[1].imshow(foreground)
    ax[1].set_title('Extracted Foreground')
    ax[1].axis('off')

    # Display the image with the contour
    ax[2].imshow(image_with_contour)
    ax[2].set_title('Image with Contour')
    ax[2].axis('off')

    # Show the plot
    plt.show()




# Usage example




