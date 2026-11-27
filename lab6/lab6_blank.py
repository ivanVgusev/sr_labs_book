#!/usr/bin/env python
# coding: utf-8

# **Лабораторный практикум по курсу «Распознавание диктора», Университет ИТМО, 2025**		

# **Лабораторная работа №6. Обучение блока построения дикторских моделей на основе трансформерных претрейнов**
# 
# **Цель работы:** изучение процедуры обучения блока построения дикторских моделей на основе трансформерных претрейнов.
# 
# **Краткое описание:** в рамках настоящей лабораторной работы предлагается изучить и реализовать схему обучения нейросетевого блока построения дикторских моделей на основе Whisper tiny трансформерного претрейна. Процедуры обучения и тестирования предлагается выполнить по отношению к задаче идентификации на закрытом множестве и верификации на открытом множестве, соответственно. Тестирование полученной системы предполагает использование доли правильных ответов (accuracy) в рамках идентификации диктора на закрытом множестве, а также равновероятной ошибки (EER, equal error rate) в рамках верификации диктора на открытом множестве в качестве целевых метрик оценки качества.
# 
# **Данные:** в качестве данных для выполнения лабораторной работы предлагается использовать базу [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).
# 
# **Содержание лабораторной работы**
# 
# 1. Подготовка данных для обучения и тестирования блока построения дикторских моделей.							
# 
# 2. Обучение параметров блока построения дикторских моделей, основанного на использовании «размороженного» Whisper tiny трансформерного претрейна.
# 
# 3. Проведение дополнительных экспериментов по использованию трансформерных архитектур в задаче распознавания диктора.

# In[ ]:


# Import of modules
import os
import sys
import itertools
from collections import OrderedDict
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
ROOT_DIR = LAB_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import download_dataset, concatenate, extract_dataset, part_extract, download_protocol, split_musan
from common import test_dataset_loader as test_dataset_loader_verification
from common import extract_features, compute_scores
from common import get_eer
from exercises_blank import train_dataset_loader, test_dataset_loader, train_network, test_network, tar_imp_hists
from exercises_blank import WhisperEncoder, TDNNSimple, StatPoolLayer, MaxoutSegmentLevel, SSLDownstream, MainModel
from whisper_fbanks import ExtractWhisperFbanks80
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer, AdamOptimizer
from Scheduler import OneCycleLRScheduler
from load_save_pth import saveParameters, loadParameters


# **1. Подготовка данных для обучения и тестирования детектора речевой активности**
# 
# В ходе выполнения лабораторной работы необходимы данные для проведения процедуры обучения и процедуры тестирования нейросетевого блока генерации дикторских моделей. Возьмём в качестве этих данных звукозаписи, сохраненные в формат *wav*, из корпуса [VoxCeleb1 dev set и VoxCeleb1 test set](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html). Данные корпуса содержат 148,642 и 4,874 звукозаписи (частота дискретизации равна 16кГц) с общим количеством дикторов женского и мужского пола равным 1,251, разговаривающих преимущественно на английском языке. В рамках настоящей лабораторной работы данные из корпуса VoxCeleb1 dev set  будут использоваться при проведения процедур обучения и валидации блока построения дикторских моделей в задаче идентификации на закрытом множестве. Данные из корпуса VoxCeleb1 test set будут использоваться для проведения верификационного теста на открытом множестве из 40 дикторов, которые отсутствовали в корпусе VoxCeleb1 dev set.
# 
# В рамках настоящего пункта требуется выполнить следующее:
# 
# 1. Загрузить и распаковать звуковые wav-файлы из корпуса VoxCeleb1 dev set и VoxCeleb1 test set.
# 
# 2. Загрузить идентификационный протокол, позволяющий разделить данные из корпуса VoxCeleb1 dev set на обучающее и валидацилонное множества.
# 
# 3. Загрузить верификационный протокол, связанный с корпусом VoxCeleb1 test set.
# 
# 4. Загрузить и распаковать данные из корпусов [SLR17 (MUSAN)](https://openslr.org/17/) and [SLR28 (RIR noises)](https://openslr.org/28/), с использованием которых можно расширить обучающее множество данных через запуск процедур аугментации.
# 
# ![Рисунок 1](../images/voxceleb_dataset.png "VoxCeleb. Крупномасштабная аудиовизуальная база данных человеческой речи.")

# In[ ]:


# Use the locally extracted VoxCeleb1 dev/test sets and protocols
from common.voxceleb_paths import (
    VOX1_DEV_WAV,
    VOX1_IDEN_SPLIT,
    VOX1_TEST_WAV,
    VOX1_VERI_TEST2,
    require_voxceleb,
)

require_voxceleb(VOX1_DEV_WAV, VOX1_TEST_WAV, VOX1_IDEN_SPLIT, VOX1_VERI_TEST2)
print(f"VoxCeleb1 dev: {VOX1_DEV_WAV}")
print(f"VoxCeleb1 test: {VOX1_TEST_WAV}")
print(f"VoxCeleb1 identification protocol: {VOX1_IDEN_SPLIT}")
print(f"VoxCeleb1 verification protocol: {VOX1_VERI_TEST2}")


# In[ ]:


# VoxCeleb data location is configured in common.voxceleb_paths.


# In[ ]:


# VoxCeleb data location is configured in common.voxceleb_paths.


# In[ ]:


# VoxCeleb data location is configured in common.voxceleb_paths.


# In[ ]:


# MUSAN and RIR data is already prepared under data.


# In[ ]:


# MUSAN and RIR archives are already extracted.


# In[ ]:


# MUSAN is already split for random access.


# **2. Обучение параметров блока построения дикторских моделей без учёта процедуры аугментации данных**
# 
# Построение современных дикторских моделей, как правило, выполняется с использованием нейросетевых архитектур. В последние годы наилучшие результаты демонстрируют [трансформерные модели](https://arxiv.org/pdf/1706.03762), существенно превосходящие по качеству более ранние свёрточные нейросетевые архитектуры. В рамках настоящего пункта предлагается выполнить адаптацию нейросетевой архитектуры [Whisper tiny](https://arxiv.org/pdf/2212.04356) под решение задачи генерации дикторских моделей (дикторских эмбеддингов). *Дикторский эмбеддинг* – это высокоуровневый вектор-признаков, состоящий, например, из 128, 256 и т.п. значений, содержащий особенности голоса конкретного человека. При решении задачи распознавания диктора можно выделить эталонные и тестовые дикторские эмбеддинги. *Эталонные эмбеддинги* формируются на этапе регистрации дикторской модели определённого человека и находятся в некотором хранилище данных. *Тестовые эмбеддинги* формируются на этапе непосредственного использования системы голосовой биометрии на практике, когда некоторый пользователь пытается получить доступ к соответствующим ресурсам. Система голосовой биометрии сравнивает по определённой метрике эталонные и тестовые эмбеддинги, формируя оценку сравнения, которая, после её обработки блоком принятия решения, позволяет сделать вывод о том, эмбеддинги одинаковых или разных дикторов сравниваются между собой.
# 
# Необходимо отметить, что построение дикторских моделей, как правило, требует наличия *акустических признаков*, вычисленных для звукозаписей тренировочной, валидационной и тестовой баз данных. В качестве примера подобных признаков в рамках настоящей лабораторной работы воспользуемся *логарифмами энергий на выходе мел-банка фильтров*. Важно отметить, что оригинальная модель Whisper tiny обучалась на 80-мерных признаках, причём входной сигнал перед обработкой путём дополнения с краю приводится к фиксированной длительности равной 30 секундам. В рамках настоящей лабораторной работы дополнение сигнала до 30 секунд не выполняется. Дополнительно, перед вычислением акустических признаков, осуществляется применение [нормализации экземпляра](https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html) (instance normalization), которая выравнивает амплитуды всех сигналов в тренировочной выборке, стабилизируя процедуру сходимости нейросетевой модели, выполняющей построение дикторских моделей, на этапе обучения. Подобная нормализация отсутствует в процедуре обучения оригинальных моделей семейства Whisper.
# 
# После того, как акустические признаки подготовлены, они могут быть переданы на блок построения дикторских моделей. Как правило, устройство современных дикторских моделей соответствует структуре [x-векторных архитектур](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf). Эти архитектуры состоят из четырёх ключевых элементов: 
# 
# 1. **Фреймовый уровень.** Предназначен для формирования локальных представлений голоса конкретного человека. На этом уровне как раз и могут быть использованы нейросетевые архитектуры на базе трансормерных моделей. При организации фреймового уровня в рамках настоящей лабораторной работы предлагается использовать предобученный кодировщик трансформерной нейросетевой архитектуры Whisper tiny, а также два [нейросетевых блока с временной задержкой](https://www.cs.toronto.edu/~hinton/absps/waibelTDNN.pdf) (TDNN, time-delay neural betwork), которые выполняют согласование кодировщика Whisper tiny с оставшейся частью нейронной сети. Использование мощных, предварительно обученных на большом объёме размеченных или неразмеченных данных трансформерных моделей, таких как [wav2vec 2.0](https://arxiv.org/pdf/2006.11477), [HuBERT](https://arxiv.org/pdf/2106.07447), [Whisper](https://arxiv.org/pdf/2212.04356) и т.п., значительно упрощает решение множества разнообразных *целевых задач* (downstream tasks), связанных с обработкой аудиосигналов, ускоряя процедуру сходимости на этапе обучения и улучшая качество работы моделей в процессе тестирования относительно моделей, обученных из случайной начальной инициализации.
# 
# 2. **Уровень статистического пулинга** позволяет сформировать промежуточный вектор-признаков, фиксированной длины, которая является одинаковой для звукозаписи любой длительности. В ходе работы блока статистического пулинга происходит удаление временной размерности, присутствующей в картах-признаков. Это достигается путём выполнения процедуры усреднения карт-признаков вдоль оси времени. Выходом уровня статистического пулинга являются вектор среднего и вектор среднеквадратического отклонения, вычисленные на основе карт-признаков. Эти вектора конкатенируются и передаются для дальнейшей обработки на сегментом уровне.
# 
# 3. **Сегментный уровень.** Предназначен для трансформации промежуточного вектора, как правило, высокой размерности, в компактный вектор-признаков, представляющий собой дикторский эмбеддинг. Необходимо отметить, что на сегментном уровне расположены один или несколько полносвязных нейросетевых слоёв, а обработка данных выполняется по отношению ко всей звукозаписи, а не только к некоторому её локальному контексту, как на фреймовом уровне.
# 
# 4. **Уровень выходного слоя.** Представляет полносвязный слой с softmax-функциями активации. Количество активаций равно числу дикторов в тренирочной выборке. На вход выходноя слоя подаётся дикторский эмбеддинг, а на выходе – формируется набор апостериорных вероятностей, определяющих принадлежность эмбеддинга к одному из дикторских классов в тренировочной выборке. Необходимо отметить, что, как правило, в современных нейросетевых системах построения дикторских моделей выходной слой используется только на этапе обучения параметров и на этапе тестирования не применяется. На этапе тестирования используются только три первых уровня архитектуры.
# 
# Обучение модели генерации дикторских эмбеддингов выполняется путём решения задачи *классификации* или, выражаясь терминами из области биометрии, *идентификации на закрытом множестве* (количество дикторских меток является строго фиксированным). В качестве используемой стоимостной функции выступает *категориальная кросс-энтропия*. Обучение выполняется с помощью мини-батчей, содержащих короткие фрагменты карт акустических признаков (длительностью несколько секунд) различных дикторов из тренировочной базы данных. Обучение на коротких фрагментах позволяет избежать сильного переобучения нейросетевой модели. При выполнении процедуры обучения требуется подобрать набор гиперпараметров, выбрать обучения и метод численной оптимизации. Обучение нейросетевых моделей при наличие претрейнов добавляет определённую специфику в процедуру обучения: использование более низкого значения параметра сходимости в случае, если вся нейросетевая модель *«разморожена»* (обучаются все веса модели на этапе тренировки), *«заморозка»* предобученной части модели (обучается часть весов модели на этапе тренировки) для ускорения процедуры сходимости и быстрой проверки гипотез и т.п.
# 
# Для успешного выполнения настоящего пункта необходимо сделать следующее:
# 
# 1. Сгенерировать списки тренировочных и валидационных данных на основе идентификационного протокола базы VoxCeleb1, заданного константой `VOX1_IDEN_SPLIT` в `common.voxceleb_paths`. При генерации списков требуется исключить из них звукозаписи дикторов, которые входят в базу [VoxCeleb1 test set](https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip). Это позволит выполнить тестирования обученных блоков генерации дикторских моделей на протоколе [VoxCeleb1-O cleaned](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt), который составлен по отношению к данным из VoxCeleb1 test set и позволяет проверить качество работы биометрической системы в рамках открытой задачи верификации диктора.
# 
# 2. Инициализировать обучаемый экстрактор дикторских эмбеддингов и вгрузить в его фреймовый уровень веса предобученного кодировщика модели Whisper tiny. Необходимо отметить, что Whisper tiny не является единственной моделью семейства [Whisper](https://github.com/openai/whisper/tree/main). Так же в общедоступных источниках могут быть найдены веса предобученных моделей Whisper base/small/medium/large. Однако, вычислительная стоимость этих моделей в сравнении с Whisper tiny является выше при более высокой точности. Для достижения более высокого качества экстрактора дикторских эмбеддингов предлагается разморозить слои кодировщика Whisper tiny на этапе обучения. 
# 
# 3. Инициализировать загрузчики тренировочной и валидационной выборок, используемые в рамках решения задачи идентификации на закрытом множестве, а также тестовой выборки, для оценки ошибки системы распознавания в рамках решения задачи верификации.
# 
# 4. Инициализировать оптимизатор и планировщик для выполнения процедуры обучения.
# 
# 5. Описать процедуру валидации блока построения дикторских моделей.
# 
# 6. Описать процедуру обучения и запустить её, контролируя значения стоимостной функции и доли правильных ответов на тренировочном и валидационном множествах, а также величину равновероятной ошибки при решении задачи верификации на тестовом множестве.

# In[ ]:


# Select hyperparameters

device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nOut              = 512                                  # embedding size

# Loss function for angular losses
margin            = 0.35                                 # margin parameter
scale             = 32.0                                 # scale parameter

# Train dataloader (close-set identification)
max_frames_train  = 400                                  # number of frame to train
train_path        = VOX1_DEV_WAV                        # path to train wav files
batch_size_train  = 128                                  # batch size to train
pin_memory        = False                                # pin memory
num_workers_train = 0                                    # number of workers to train
shuffle           = True                                 # shuffling of training examples

# Validation dataloader (close-set identification)
max_frames_val    = 400                                  # number of frame to validate
val_path          = VOX1_DEV_WAV                        # path to val wav files
batch_size_val    = 128                                  # batch size to validate
num_workers_val   = 0                                    # number of workers to validate

# Test dataloader (open-set verification)
max_frames_test   = 400                                  # number of frame to test
test_path         = VOX1_TEST_WAV                       # path to val wav files
batch_size_test   = 1                                    # batch size to test
num_workers_test  = 0                                    # number of workers to test
num_eval          = 5                                    # number of enroll/test chunks for scoring

# Optimizer
lr                = 0.0001                               # learning rate value # 0.0001
weight_decay      = 0                                    # weight decay value

# Scheduler
val_interval      = 5                                    # frequency of validation step
max_epoch         = 40                                   # number of epoches

# Augmentation
musan_path        = DATA_DIR / 'musan_split'                # path to splitted SLR17 dataset
rir_path          = DATA_DIR / 'RIRS_NOISES/simulated_rirs' # path to SLR28 dataset

for required_path in (musan_path, rir_path):
    if not required_path.is_dir():
        raise FileNotFoundError(f'Missing augmentation data: {required_path}')


# In[ ]:


# Generate train/validation data lists for close-set speaker identification
train_list = []
val_list   = []

with open(VOX1_IDEN_SPLIT, 'r') as f:
    lines = f.readlines()
    
black_list = os.listdir(VOX1_TEST_WAV)                  # exclude speaker IDs from VoxCeleb1 test set
num_train_spk = []                                      # number of train speakers

for line in lines:
    line   = line.strip().split(' ')
    spk_id = line[1].split('/')[0]
    
    if not (spk_id in black_list):
        num_train_spk.append(spk_id)
        
    else:
        continue
    
    # Train list
    if (line[0] == '1'):
        train_list.append(' '.join([spk_id, line[1]]))
    
    # Validation list
    elif (line[0] == '2'):
        val_list.append(' '.join([spk_id, line[1]]))
        
num_train_spk = len(set(num_train_spk))


# In[ ]:


# Initialize train dataloader (with augmentation)
train_dataset = train_dataset_loader(train_list=train_list, 
                                     max_frames=max_frames_train, 
                                     train_path=train_path, 
                                     augment=True, 
                                     musan_path=musan_path, 
                                     rir_path=rir_path)

train_loader  = DataLoader(train_dataset, 
                           batch_size=batch_size_train, 
                           pin_memory=pin_memory, 
                           num_workers=num_workers_train, 
                           shuffle=shuffle)

# Initialize validation dataloader
val_dataset = test_dataset_loader(test_list=val_list, max_frames=max_frames_val, test_path=val_path)
val_loader  = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers_val)


# In[ ]:


# Load test protocol
with open(VOX1_VERI_TEST2, 'r') as f:
    verif_protocol_lines = f.readlines()
    
# Get a list of unique file names
files = list(itertools.chain(*[x.strip().split()[-2:] for x in verif_protocol_lines]))
setfiles = list(set(files))
setfiles.sort()

# Initialize test dataloader
test_dataset = test_dataset_loader_verification(setfiles, test_path=test_path, eval_frames=max_frames_test, num_eval=num_eval)
test_loader  = DataLoader(test_dataset, 
                          batch_size=batch_size_test, 
                          shuffle=False, 
                          num_workers=num_workers_test, 
                          drop_last=False, 
                          sampler=None)


# In[ ]:


# Initialize model
preprocessor        = ExtractWhisperFbanks80(pad_data=False)
feat_extractor      = WhisperEncoder.build_model(model_name="tiny", load_weight=True, n_layer=-1) # Whisper tiny download to 
                                                                                                  # ~/.cache/whisper/tiny.pt
frame_level_block   = TDNNSimple(n_layers=1, in_ch=384, out_ch=1024)
pooling_block       = StatPoolLayer.by_string_mode(mode="MV")
segment_level_block = MaxoutSegmentLevel(input_dim=2048, output_dim=512, enable_batch_norm=True)

whisper_sr_model    = SSLDownstream(preproc=preprocessor,
                                    feat_extractor=feat_extractor,
                                    frame_level_block=frame_level_block,
                                    pooling_block=pooling_block,
                                    segment_level_block=segment_level_block,
                                    freeze_feats=False)
trainfunc           = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
main_model          = MainModel(whisper_sr_model, trainfunc).to(device)


# In[ ]:


# Initialize optimizer and scheduler
optimizer = SGDOptimizer(main_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = OneCycleLRScheduler(optimizer, 
                                pct_start=0.05, 
                                cycle_momentum=False, 
                                max_lr=lr, 
                                div_factor=10, 
                                final_div_factor=10000, 
                                total_steps=max_epoch*len(train_loader))


# In[ ]:


checkpoint_dir = DATA_DIR / 'lab6_models'
latest_checkpoint = checkpoint_dir / 'lab6_latest.pth'
numbered_checkpoints = sorted(checkpoint_dir.glob('lab6_model_*.pth'))
checkpoint_path = latest_checkpoint if latest_checkpoint.exists() else None
if checkpoint_path is None and numbered_checkpoints:
    checkpoint_path = numbered_checkpoints[-1]

start_epoch = 0
if checkpoint_path is not None:
    start_epoch = loadParameters(main_model, optimizer, scheduler, path=checkpoint_path)
    start_epoch = start_epoch + 1
    print(f'Resuming training from epoch {start_epoch}: {checkpoint_path}')

# Train model
for num_epoch in range(start_epoch, max_epoch):
    train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=True)
    
    print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))

    saveParameters(
        main_model,
        optimizer,
        scheduler,
        num_epoch,
        path=checkpoint_dir,
        filename='lab6_latest.pth',
    )

    if (num_epoch + 1)%val_interval == 0:
        _, val_top1 = test_network(val_loader, main_model)                               # compute accuracy for validation set
        
        print("Epoch {:1.0f}, Accuracy (validation set) {:2.3f}%".format(num_epoch, val_top1))
        
        feats = extract_features(main_model, test_loader)                                # compute enroll/test speaker embeddings
        all_scores, all_labels, all_trials = compute_scores(feats, verif_protocol_lines) # compute scores between enroll and test speaker models
        tar_scores, imp_scores = tar_imp_hists(all_scores, all_labels)                   # devide the scores into target and impostor scores
        EER, _ = get_eer(tar_scores, imp_scores)                                         # compute EER, %
        
        print("Epoch {:1.0f}, EER (test set) {:2.3f}%".format(num_epoch, EER))
        
        saveParameters(main_model, optimizer, scheduler, num_epoch, path=checkpoint_dir)


# **3. Дополнительные задания по обучению дикторской модели на основе трансформерной архитектуры**
# 
# В рамках настоящего пункта предлагается выбрать и решить одно из дополнительных заданий на выбор:
# 
# 1. Переведя параметр load_weight кодировщика Whisper tiny в режим False, попробуйте обучить трансформерную нейронную сеть без предварительно загруженных в неё весов кодировщика. Сравните полученное качество модели с тем, что было в пункте 2. При проведении исследования разрешается менять любые настройки процедуры обучения, в частности, параметр сходимости.
# 
# 2. Переведя параметр freeze_feats кодировщика Whisper tiny в режим True, попробуйте обучить трансформерную нейронную сеть с «замороженным» кодировщиком. Сравните полученное качество модели с тем, что было в пункте 2. При проведении исследования разрешается менять любые настройки процедуры обучения, в частности, параметр сходимости.
# 
# 3. В настройках процедуры обучения рассогласуйте параметр, отвечающий за максимальное число фреймов, используемых на этапе обучения, валидации и тестирования, например, выставив его равным 400 на этапе обучения и 1000 на этапах валидации и тестирования. Попробуйте обучить трансформерную нейронную сеть для выбранных настроек. Сравните полученное качество модели с тем, что было в пункте 2.
# 
# 4. Попробуйте заменить кодировщика Whisper tiny на Whisper модель больших размеров base/small/medium/large и повторите процедуру обучения из пункта 2. Сравните полученное качество модели с тем, что было в пункте 2.

# **4. Контрольные вопросы**
# 
# 1. Что такое трансформерные нейросетевые модели и для какой задачи они были изначально предложены?
# 
# 2. Назовите основные отличия трансформерных нейросетевых архитектур от свёрточных?
# 
# 3. Как устроена архитектура Whisper и для решения каких задач она была изначально разработана?
# 
# 4. Из каких основных нейросетевых слоёв состоит архитектура Whisper?
# 
# 5. Какие разновидности архитектуры Whisper существуют и чем они отличаются друг от друга?
# 
# 6. Что такое предобученная нейросетевая модель и для каких целей она используется?
# 
# 7. В чём состоит основное отличие в политике изменения параметра сходимости на этапе обучения при наличии преобученной модели от ситуации, когда она отсутствует? 
# 
# 8. Назовите примеры трансформерных нейросетевых претрейнов для обработки аудиосигналов и кратко опишите, как они были получены?
# 
# 9. Как реализовать блок построения дикторских моделей на основе архитектуры Whisper?

# **5. Список литературы**
# 
# 1. Bai Z., Zhang X.-L., Chen J. Speaker recognition based on deep learning: an overview // 	arXiv:2012.00931 [eess.AS] ([ссылка](https://arxiv.org/pdf/2012.00931.pdf)).
# 
# 2. Hansen J.H.L., Hasan T. Speaker recognition by machines and humans: a tutorial review // IEEE Signal Processing Magazine, 2015. V. 32. № 6. P. 74–99 ([ссылка](https://www.researchgate.net/publication/282940395_Speaker_Recognition_by_Machines_and_Humans_A_tutorial_review)).
# 
# 3. Vaswani A. et al. Attention is all you need // arXiv:1706.03762 [cs.CL] ([ссылка](https://arxiv.org/pdf/1706.03762)).
# 
# 4. Radford A. et al. Robust speech recognition via large-scale weak supervision // arXiv:2212.04356 [eess.AS] ([ссылка](https://arxiv.org/pdf/2212.04356)).
