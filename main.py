import argparse
import cv2
import mediapipe as mp

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='webcam', help="Mode of operation: image, video, or webcam")
    parser.add_argument("--filePath", default=None, help="Path to the image or video file")

    args = parser.parse_args()

    # Detect faces
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == "image":
            if args.filePath is None:
                print("File path is required for image mode")
                return
            # Read image
            img = cv2.imread(args.filePath)
            if img is None:
                print("Could not read the image")
                return

            img = process_img(img, face_detection)

            # Display image
            cv2.imshow('Processed Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Image processed and displayed")

        elif args.mode == 'video':
            if args.filePath is None:
                print("File path is required for video mode")
                return
            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            if not ret:
                print("Could not read the video")
                return

            while ret:
                frame = process_img(frame, face_detection)
                cv2.imshow('Processed Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()
            print("Video processed and displayed")

        elif args.mode == 'webcam':
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Could not open webcam")
                return

            ret, frame = cap.read()
            while ret:
                frame = process_img(frame, face_detection)
                cv2.imshow('Face Detecting and Blurring', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()
            print("Webcam processing terminated")


if __name__ == "__main__":
    main()
