import cv2

video_path = r'C:\Users\HP\Desktop\testing\data\videos\pizza.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Can't read video")
    exit()

drawing = False
ix, iy = -1, -1
ROIs = []

def draw(event, x, y, flags, param):
    global ix, iy, drawing, ROIs
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ROIs.append((ix, iy, x, y))
        print(f"ROI: ({ix}, {iy}, {x}, {y})")

cv2.namedWindow("Select ROIs")
cv2.setMouseCallback("Select ROIs", draw)

while True:
    temp = frame.copy()
    for (x1, y1, x2, y2) in ROIs:
        cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imshow("Select ROIs", temp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final ROIs:")
for roi in ROIs:
    print(roi)
