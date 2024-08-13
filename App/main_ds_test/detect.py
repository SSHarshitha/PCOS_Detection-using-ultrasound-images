from ultralytics import YOLO
import shutil

model = YOLO(r"C:\Users\Nithya Sri\Downloads\PCOS Detect\APP\main_ds_test\best.pt")

def detect():
    
    # Make predictions
    src = "static\\uploaded_files\\image.jpg"
    results = model.predict(source=src, show=True, save=True, conf=0.4, save_dir = "preds", line_thickness=1)
    
    # Access information from the results dictionary
    for item in results:
        #print(vars(item))
        #print (item.save_dir)
        #print(len(item.boxes))

        source_path = item.save_dir + "\\image.jpg"
        destination_path = "static\\uploaded_files\\out_image.jpg"
        shutil.copy(source_path, destination_path)
        return(len(item.boxes))

if __name__ == '__main__':
    print(detect())