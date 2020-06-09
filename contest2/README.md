This pipeline consists of detection through segmentation + recognition.
Python 3.7.6, Pytorch 1.4.0

For segmentation/recognition datasets you've got to run create_datasets.ipynb to convert bboxes to segmentation masks and crop them for OCR.

Each task can be trained using train.py in the respective dir. All the parameters, used to obtain the baseline,
 are set to default, except data_path.

Once the 2 models are trained (you can find the baseline models in pretrained sections), you can proceed to inference.py and generate a submission. 

Current networks and routines are basic, there's plenty of room for improvement and quick wins.
 The "TODO" sections are scattered across the code and hint to the possible problems/avenues of research, but TODOs are 
 not extensive, so do not rely solely on them.

The pipeline is by no means optimal, you can extend it by introducing new components (neural networks/heuristics),
 especially between the two current networks.
 
Also the baseline overfits and doesn't score good on the private part of the test dataset, which comes from a bit different distribution. 

Good luck and have fun!


PS. Some unusual requirements:
 - pip install editdistance


# Отчет
1) Прикрутил Tensorboard
    - В первую очередь добавил логгирование различных значений
    loss'ов и lr для batch/epoch в train/val режимах
        - чтобы лучше видеть пере-/недообучение
        - надеюсь, что это поможет подобрать `lr` и параметры для `scheduler`'а
    - Для segmentation разделил loss на `BCE` и `Dice`(складываю с весами в самом конце)
        - думаю, что так будет легче понять, с какими весами их лучше складывать в итоговый loss
2) Погуглил существующие подходы к решению задачи + изучил бэйзлайны однокурсников. 
Очень понравился бэйзлайн <a href="https://www.kaggle.com/alyar88/maskrcnn-bb-x-mask-crnn-0-53">alexyar88</a>.
    - Хорошая идея с использованием `Mask RCNN`. Так как модель обучается и на предсказание bounding box'а с номером внутри, 
    и на маскирование самого номера, то она обладает большей обобщающей способностью, что позволяет точней решать каждую из задач
    - Понравилась идея с преобразованием четырехугольника с номером в прямоуглоьник(эта идея была у многих)
    - Интересное усреднение предсказаний `OCR` для bbox'а с номером и для номера, преобразованного в прямоугольник
3) Времени из-за приближающеся защиты диплома было немного, так что решения лучше я бы вряд ли придумал, 
поэтому решил перенести описанные выше идеи в преподавательский бэйзлайн:
    - заменил модель детектирования на `Mask RCNN`
    - добавил подготовку данных для `Mask RCNN`
    - добавил функции, реализованные `alexyar88`
    - добавил в формирование датасета для `OCR` bbox и "выпрямленный" вариант номера
    - добавил `AvgPooling` в `OCR`, чтобы увеличить высоту номерного знака(полагаю, что это помогло понизить переобучение) 
    - поменял сохранение `OCR` модели, чтобы подгрузать параметры модели
    - адаптировал `inference.py` под новый пайплайн
4) Обучил модели:
    - `Mask_RCNN_v1`: 
        - batch_size: `10`
        - epochs: `1`
        - train_size: `0.95`
        - val: (упала)
    - `OCR_v1`:
        - backbone: `resnet18`
        - batch_size: `128`
        - epochs: `4`
        - train_size: `0.95`
        - val: 
            - acc_ed: `0.094`
            - acc: `0.942`
5) Сабмиты:
    1) `Mask_RCNN_v1` + `OCR_v1`:
        - Public score: `0.596`