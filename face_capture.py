import cv2

def capture():
	cam_port = 0
	cam = cv2.VideoCapture(cam_port)

	result, image = cam.read()

	if result:
		cv2.imshow("Face Recognition", image)
		cv2.imwrite("user.png", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	else:
		print("No image detected. Please! try again")

if __name__ == '__main__':
    capture()