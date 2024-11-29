import time
start_time = time.perf_counter()


import os, os.path
import glob

from ultralytics import ASSETS, YOLO
import supervision as sv
from PIL import Image


previewOutput = False
img_dir = "C:/Users/Matt/Pictures/Input"
outputDir = "C:/Users/Matt/Pictures/Output"


image_list = [item for i in [glob.glob(f'{img_dir}/*.%s' % ext) for ext in ["jpg","gif","png","tga"]] for item in i]
# load model
model = YOLO('weights/best.pt', task="detect")
loading_time = time.perf_counter()
onnxFile = model.export(format="onnx")
print("\n----------------------------------")
print(f"\nTook {(loading_time - start_time):.3f}s to load Dependencies and Model.")

i = 0
for img_path in image_list:

    image_processing_start_time = time.perf_counter()

    #img_path = "datasets/bottles/valid/images/-169_jpg.rf.ea758cf080a3a90b71a11c0f025c398d.jpg"
   
    #print(model)
    #results = model(source = ASSETS / img_path)
    results = model.predict(img_path, conf=0.5)

    # supervision
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    image = Image.open(img_path)
    annotated_image = image.copy()

    for result in results:
        #result.print()
        #labels, cord_thres = result.xyxyn[0][:,-1].numpy(), result.xyxyn[0][:,:-1].numpy()
        if previewOutput:
            print(result.boxes.data)
        #image = result.show()

        detections = sv.Detections.from_ultralytics(result)
        labels = [model.model.names[class_id]
            for class_id
            in detections.class_id]

        j = 0
        for label in labels:
            w = detections.xyxy[j][2] - detections.xyxy[j][0]
            h = detections.xyxy[j][3] - detections.xyxy[j][1]

            label = f'{label}, {detections.confidence[j]:.2f}, W: {w:.0f}, H: {h:.0f}'

            # update labels
            labels[j] = label

            j+=1
            
        annotated_image = box_annotator.annotate(annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)

    if previewOutput:
        sv.plot_image(annotated_image, size=(10, 10))

    #with sv.ImageSink(target_dir_path="output") as sink:
    #    sink.save_image(image=annotated_image)#, image_name=f"output_{i}.jpg")
    annotated_image.convert('RGB').save(f"{outputDir}/{i}.jpg")

    image_processing_end_time = time.perf_counter()
    print(f"Took {(image_processing_end_time - image_processing_start_time):.3f}s to load process image {i+1}.")

    i += 1

current_time = time.perf_counter()
print(f"\n\nTook {(current_time - start_time):.3f}s to complete,")
print(f"of which {(loading_time - start_time):.3f}s was loading time,")
print(f"giving {(current_time - loading_time):.3f}s to process {i} item(s).\n\n")
