if __name__ == '__main__':
    from ultralytics import YOLO
    import cv2

    # Load the model
    model = YOLO("model.pt")  #

    # Test the model
    results = model("111.jpg", device = 0, show = True)
    cv2.waitKey(0)



