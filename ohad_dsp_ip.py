
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import ImageColor, ImageFont, Image, ImageDraw

def draw_results(image, results, relevant_class_id =[2]):
  """Draw the detection results on a copy of the source image"""

  colors = list(ImageColor.colormap.values())
  new_image = image.copy()
  font = ImageFont.load_default()

  for i in range(len(results.xyxy[0])):
    det_line = results.xyxyn[0].numpy()[i]
    boxes = det_line[:4]
    score = det_line[4]
    class_id = det_line[5]
    class_name = results.names[class_id]

    if class_id in relevant_class_id:
      xmin, ymin, xmax, ymax  = boxes

      display_str = "{}: {}%".format(class_name,
                                      int(100 * score))
      
      color = colors[hash(class_name) % len(colors)]
      image_pil = Image.fromarray(np.uint8(new_image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(new_image, np.array(image_pil))
  return new_image

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=2,
                               display_str_list=()):
  
  """Adds a bounding box and class name text box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    
    text_bottom -= text_height - 2 * margin

def res_to_list(results, frame_num, res_list=[]):
    frame_res = results.xyxyn[0].numpy()
    obj_count = frame_res.shape[0]

    for j in range(obj_count):
        line = [frame_num, j+1] + frame_res[j].tolist()
        res_list.append(line)
    return res_list


def main():

    H = 360
    W = 640
    fps = 20
    model_version = 'yolov5m'
    PATH = 'data/fpv-drone-vs-rallycross-cars.mp4'
    JSON_OUT_PATH = 'detection_output.json'
    OUTPUT_PATH = 'output/output_vid.mp4'
    CONF_THRSHOLD = 0.73

    # load model
    model = torch.hub.load('ultralytics/yolov5', model_version, pretrained=True)
 
    # set confidence threshold
    model.conf = CONF_THRSHOLD  
    # model.iou = 0.25  # NMS IoU threshold

    # capture video
    cap = cv2.VideoCapture(PATH)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W,H))

    result_list = []
    while True:
    # for i in range(0,200,20):
        ret, original_frame = cap.read()
        if ret==True:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
  
            # get detection results
            results = model(frame)

            # update results list 
            result_list = res_to_list(results, i, result_list)
            # draw results on image 
            image_with_boxes = draw_results(frame, results)

            # write the output frame
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
            out.write(image_with_boxes)

        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()

    # create df from the result_list and convert it to JSON
    col_list = ['frame', 'object', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
    df = pd.DataFrame(result_list, columns = col_list)
    df.to_json(JSON_OUT_PATH, orient="index", indent=2) 

if __name__ == "__main__":
    main()
