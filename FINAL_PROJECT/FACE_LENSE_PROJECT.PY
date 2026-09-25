!pip install -q ultralytics "deepface[tensorflow]" tf-keras mtcnn

import cv2, io, numpy as np
from collections import deque, Counter
from base64 import b64decode, b64encode
from PIL import Image as PILImage
from IPython.display import display, Javascript
from google.colab.output import eval_js
from ultralytics import settings
settings.update({"sync": False})
from ultralytics import YOLO
from deepface import DeepFace

yolo_model = YOLO('yolov8s.pt')   # CHANGED: was yolov8n.pt (nano) — 's' is more accurate, a bit slower

def video_stream():
  display(Javascript('''
    var video, div = null, stream, captureCanvas, imgElement, labelElement;
    var pendingResolve = null, shutdown = false;

    function removeDom() {
      stream.getVideoTracks()[0].stop();
      video.remove(); div.remove();
      video = null; div = null; stream = null; imgElement = null;
      captureCanvas = null; labelElement = null;
    }
    function onAnimationFrame() {
      if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8);
        }
        var lp = pendingResolve; pendingResolve = null; lp(result);
      }
    }
    async function createDom() {
      if (div !== null) return stream;
      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);

      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status: </span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'Starting...';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);

      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia({video: true});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.innerHTML = '<span style="color:red;font-weight:bold;">Click here or on the video to STOP</span>';
      instruction.onclick = () => { shutdown = true; };
      div.appendChild(instruction);

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640;
      captureCanvas.height = 480;
      window.requestAnimationFrame(onAnimationFrame);
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) { removeDom(); shutdown = false; return ''; }
      stream = await createDom();
      if (label != "") labelElement.innerHTML = label;
      if (imgData != "") {
        var r = video.getClientRects()[0];
        imgElement.style.top = r.top + "px";
        imgElement.style.left = r.left + "px";
        imgElement.style.width = r.width + "px";
        imgElement.style.height = r.height + "px";
        imgElement.src = imgData;
      }
      var result = await new Promise(function(resolve) { pendingResolve = resolve; });
      shutdown = false;
      return {'img': result};
    }
  '''))

def video_frame(label, overlay):
  return eval_js('stream_frame("{}", "{}")'.format(label, overlay))

def js_to_image(js_reply):
  img_bytes = b64decode(js_reply.split(',')[1])
  return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

def overlay_to_data_url(rgba):
  buf = io.BytesIO()
  PILImage.fromarray(rgba, 'RGBA').save(buf, format='png')
  return 'data:image/png;base64,' + b64encode(buf.getvalue()).decode()

video_stream()
label = 'Starting...'
overlay = ''

FACE_EVERY_N = 12
frame_no = 0
faces_info = []
age_history = deque(maxlen=15)
emotion_history = deque(maxlen=10)
gender_history = deque(maxlen=10)   # CHANGED: added for gender smoothing

while True:
  reply = video_frame(label, overlay)
  if not reply:                       # click the video to stop
    break

  frame = js_to_image(reply['img'])
  frame_no += 1

  # --- object detection (every frame) ---
  results = yolo_model(frame, verbose=False)
  boxes = results[0].boxes
  detected_count = len(boxes)

  layer = np.zeros((480, 640, 4), dtype=np.uint8)
  for b in boxes:
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    name = yolo_model.names[int(b.cls[0])]
    conf = float(b.conf[0])
    cv2.rectangle(layer, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)
    cv2.putText(layer, f'{name} {conf:.2f}', (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0, 255), 2, cv2.LINE_AA)

  # --- age + gender + emotion, for ALL faces ---
  if frame_no % FACE_EVERY_N == 0:
    try:
      analysis = DeepFace.analyze(
          frame, actions=['age', 'gender', 'emotion'],
          enforce_detection=True,
          silent=True,
          detector_backend='retinaface'
      )
      new_faces = analysis if isinstance(analysis, list) else [analysis]
      new_faces = [f for f in new_faces if f['region']['w'] > 20 and f['region']['h'] > 20]

      if new_faces:
        main_face = max(new_faces, key=lambda f: f['region']['w'] * f['region']['h'])

        # smooth age with rolling median
        age_history.append(main_face['age'])
        main_face['age'] = int(np.median(age_history))

        # smooth emotion with rolling most-common value
        emotion_history.append(main_face['dominant_emotion'])
        main_face['dominant_emotion'] = Counter(emotion_history).most_common(1)[0][0]

        # CHANGED: smooth gender with rolling most-common value
        gender_history.append(main_face['dominant_gender'])
        main_face['dominant_gender'] = Counter(gender_history).most_common(1)[0][0]

      faces_info = new_faces
    except Exception:
      pass   # no confident face found this cycle; keep showing last known faces_info

  for f in faces_info:
    r = f['region']
    x1, y1, w, h = r['x'], r['y'], r['w'], r['h']
    x2, y2 = x1 + w, y1 + h
    age = f['age']
    gender = f['dominant_gender']
    mood = f['dominant_emotion']

    cv2.rectangle(layer, (x1, y1), (x2, y2), (255, 255, 0, 255), 2)
    cv2.putText(layer, f'{gender}, {age}y, {mood}', (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0, 255), 2, cv2.LINE_AA)

  cv2.putText(layer, f'Faces: {len(faces_info)}  Objects: {detected_count}', (20, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2, cv2.LINE_AA)

  overlay = overlay_to_data_url(layer)
  label = f'Faces: {len(faces_info)} | Objects: {detected_count}'

print("Camera feed closed successfully.")
  label = f'Faces: {len(faces_info)} | Objects: {detected_count}'

print("Camera feed closed successfully.")
