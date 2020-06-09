import os, sys, cv2, numpy as np
import torch, torchvision
from argparse import ArgumentParser

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from contest2.alexyar88_baseline_utils import simplify_contour, four_point_transform
from contest2.recognition.model import RecognitionModel
from contest2.recognition.transform import Resize
from contest2.utils import prepare_for_inference, get_boxes_from_mask

from contest2.segmentation.mask_rcnn import get_detector_model


def ocr_preprocess(image, output_size):
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float) / 255.
    return torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None, help='path to the data')
    parser.add_argument('-ts', '--threshold_score', dest='threshold_score', default=0.9, type=float, help='threshold score')
    parser.add_argument('-tm', '--threshold_mask', dest='threshold_mask', default=0.05, type=float, help='threshold mask')
    parser.add_argument('-s', '--seg_model', dest='seg_model', type=str, default=None,
                        help='path to a trained segmentation model')
    parser.add_argument('-r', '--rec_model', dest='rec_model', type=str, default=None,
                        help='path to a trained recognition model')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='log_inference',
                        help='dir to save log and models')
    args = parser.parse_args()
    print('Start inference')
    os.makedirs(args.output_dir, exist_ok=True)

    cur_dir = os.path.dirname(__file__)
    seg_model_path = args.seg_model or os.path.join(cur_dir, 'pretrained', 'seg_unet_20epoch_basic.pth')
    rec_model_path = args.rec_model or os.path.join(cur_dir, 'pretrained', 'rec-crnn-30epoch_basic.pth')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg_model = get_detector_model()
    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.to(device)
    seg_model.eval()
    seg_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    checkpoint = torch.load(rec_model_path)
    recognition_model = RecognitionModel(**checkpoint['model_config'])
    recognition_model.load_state_dict(checkpoint['model_state_dict'])
    recognition_model.to(device)
    recognition_model.eval()
    h, w = checkpoint['model_config']['input_size']
    resizer = Resize(size=(w, h))
    
    test_dir_path = os.path.join(args.data_path, 'test')
    results = []
    with torch.no_grad():
        files = os.listdir(test_dir_path)
        for i, file_name in tqdm(enumerate(files), total=len(files)):
            img_path = os.path.join(test_dir_path, file_name)
            img = Image.open(img_path).convert('RGB')
            img_cv = cv2.imread(img_path)
            img_tensor = seg_transforms(img)

            prediction = seg_model([img_tensor.to(device)])[0]
            texts = []
            for j in range(len(prediction['boxes'])):
                x_min, y_min, x_max, y_max = map(int, prediction['boxes'][j].tolist())
                score = float(prediction['scores'][j].cpu())
                mask = prediction['masks'][j][0, :, :].cpu().numpy()

                if score > args.threshold_score:
                    # В разных версиях opencv этот метод возвращает разное число параметров
                    # Оставил для версии colab
                    contours, _ = cv2.findContours((mask > args.threshold_mask).astype(np.uint8), 1, 1)
                    #             _,contours,_ = cv2.findContours((mask > TRESHOLD_MASK).astype(np.uint8), 1, 1)
                    approx = simplify_contour(contours[0], n_corners=4)

                    if approx is None:
                        x0, y0 = x_min, y_min
                        x1, y1 = x_max, y_min
                        x2, y2 = x_min, y_max
                        x3, y3 = x_max, y_max
                    else:
                        x0, y0 = approx[0][0][0], approx[0][0][1]
                        x1, y1 = approx[1][0][0], approx[1][0][1]
                        x2, y2 = approx[2][0][0], approx[2][0][1]
                        x3, y3 = approx[3][0][0], approx[3][0][1]

                    points = [[x0, y0], [x2, y2], [x1, y1], [x3, y3]]

                    img_bbox = resizer(img_cv[y_min:y_max, x_min:x_max])
                    img_bbox = seg_transforms(img_bbox)
                    img_bbox = img_bbox.unsqueeze(0)

                    points = np.clip(np.array(points), 0, None)
                    img_polygon = resizer(four_point_transform(img_cv, points))
                    img_polygon = seg_transforms(img_polygon)
                    img_polygon = img_polygon.unsqueeze(0)

                    preds_bbox = recognition_model(img_bbox.to(device)).cpu().detach()
                    preds_poly = recognition_model(img_polygon.to(device)).cpu().detach()

                    preds = preds_poly + preds_bbox
                    text = recognition_model.decode(preds)[0]

                    texts.append((x_min, text))
            
            # all predictions must be sorted by x1
            texts.sort(key=lambda x: x[0])
            results.append((file_name, [w[1] for w in texts]))
            
    # Generate a submission file
    with open(os.path.join(args.output_dir, 'submission.csv'), 'w') as wf:
        wf.write('file_name,plates_string\n')
        for file_name, texts in sorted(results, key=lambda x: int(os.path.splitext(x[0])[0])):
            wf.write('test/%s,%s\n' % (file_name, ' '.join(texts)))
    print('Done')
            

if __name__ == '__main__':
    sys.exit(main())
