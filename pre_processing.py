import os
import shutil
import tgt
import librosa
import numpy as np

folder_list = ['HQTV', 'PNV', 'THV', 'TLV']
root_path = 'dataset'
label_list = []
sample_rate = 16000
max_len = 32000

def create_subfolder(root_path, folder_list):
    anno_folder = 'annotation'
    wav_folder = 'wav'
    for folder in folder_list:
        folder_path = os.path.join(root_path, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        sub_folder_path = os.path.join(folder_path, anno_folder)
        if not os.path.exists(sub_folder_path):
            os.mkdir(sub_folder_path)
        sub_folder_path = os.path.join(folder_path, wav_folder)
        if not os.path.exists(sub_folder_path):
            os.mkdir(sub_folder_path)

def copy_data(root_path, folder_list):
    anno_folder = 'annotation'
    wav_folder = 'wav'
    for folder in folder_list:
        source_folder = os.path.join(folder, anno_folder)
        file_list = os.listdir(source_folder)
        for file in file_list:
            # Copy annotation files
            source_path = os.path.join(source_folder, file)
            dest_path = os.path.join(root_path, folder, anno_folder, file)
            shutil.copyfile(source_path, dest_path)
            # Copy wav files
            wav_file = file.split('.')[0] + '.wav'
            source_path = os.path.join(folder, wav_folder, wav_file)
            dest_path = os.path.join(root_path, folder, wav_folder, wav_file)
            shutil.copyfile(source_path, dest_path)

def read_all_label():
    count_error = 0
    for folder in folder_list:
        file_list = os.listdir(os.path.join(root_path, folder, 'annotation'))
        for file in file_list:
            file_path = os.path.join(root_path, folder, 'annotation', file)
            try:
                tg_file = tgt.read_textgrid(file_path)
                item = tg_file.get_tier_by_name('phones')
                for i in item:
                    text = i.text
                    if text not in label_list:
                        label_list.append(text)
            except:
                count_error +=1
                print('Processing file: ', file_path)
                continue
    print('Num of error files: ', count_error)

def process_file(textgrid_file, wav_file, sample_rate=16000):
    tg_file = tgt.read_textgrid(textgrid_file)
    wav_arr, _ = librosa.load(wav_file, sr=sample_rate)
    words = tg_file.get_tier_by_name('words')
    word_list = []
    for w in words:
        start_time = float(w.start_time)
        end_time = float(w.end_time)
        text = w.text
        word_list.append([text, start_time, end_time])

    phones = tg_file.get_tier_by_name('phones')
    phone_list = []
    for p in phones:
        start_time = float(p.start_time)
        end_time = float(p.end_time)
        text = p.text
        phone_list.append([text, start_time, end_time])

    word_len = len(word_list)
    X, y = [], []
    j = 0
    for i in range(word_len):
        w_start_time = word_list[i][1]
        w_end_time = word_list[i][2]
        count_err = 0
        count = 0
        while j< len(phone_list) and phone_list[j][1] < w_end_time:
            p_text = phone_list[j][0]
            count +=1
            j+=1
            if ',' in p_text:
                count_err+=1
        point = (count - count_err)  * 1. / count
        wav_word = wav_arr[int(w_start_time * sample_rate) : int(w_end_time * sample_rate)]
        if len(wav_word)> max_len:
            wav_word = wav_word[:max_len]
        else:
            wav_word = np.pad(wav_word, (0, max_len - len(wav_word)))
        if point >= 0.7:
            label = 3
        elif point >= 0.4:
            label = 2
        else:
            label = 1
        X.append(wav_word)
        y.append(label)

    return X, y

def processing_data():
    X_full, y_full = [], []
    for folder in folder_list:
        file_list = os.listdir(os.path.join(root_path, folder, 'annotation'))
        for file in file_list:
            textgrid_path = os.path.join(root_path, folder, 'annotation', file)
            wav_path = os.path.join(root_path, folder, 'wav', file.split('.')[0] + '.wav')
            X, y = process_file(textgrid_file=textgrid_path, wav_file=wav_path)
            X_full += X
            y_full +=y
    return X_full, y_full

X_full, y_full = processing_data()
np.savez('data.npz', X=np.array(X_full), y=np.array(y_full))
