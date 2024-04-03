import torch
from typing import List
import os
import numpy as np
import cv2
import imageio
import datetime
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
from torch.optim import Adam
from accelerate import Accelerator

class CustomDataset(Dataset): 
    def __init__(self):
        self.video_list = glob.glob('./small_data/s1/*.mpg')
        self.video_list=sorted(self.video_list)
        self.align_list = glob.glob('./small_data/alignments/s1/*.align')
        self.align_list=sorted(self.align_list)
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        frames=self.load_video(idx) 
        alignments =self.load_alignments(idx)
        assert frames.shape[0]==75, f"75 아니고 {frames.shape[0]}"
        assert frames.shape[1]==1, f"1 아니고 {frames.shape[1]}"
        assert alignments.shape[0]==40, f"1 아니고 {alignments.shape[0]}"
        return frames,alignments
    
    def load_video(self, idx):
        cap = cv2.VideoCapture(self.video_list[idx])
        frames = []
        transform = transforms.Grayscale()  # RGB to Grayscale 변환

        # 비디오의 모든 프레임을 읽어서 처리
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV에서는 BGR로 이미지를 로드하므로 RGB로 변환
            frame = torch.tensor(frame).permute(2, 0, 1)  # HWC to CHW
            frame = transform(frame.float())  # Grayscale 변환
            
            frame = frame[:, 190:236, 80:220]  # 프레임 자르기
            frames.append(frame)

        cap.release()  # 모든 프레임을 처리한 후, 비디오 캡처를 해제

        if frames:  # 프레임이 하나라도 존재하는 경우에만 처리 
            frames = torch.stack(frames)
            if frames.shape[0] < 75:
                padding = torch.zeros(75 - frames.shape[0],frames.shape[1],frames.shape[2],frames.shape[3])
                frames=torch.cat((frames, padding), dim=0)
            mean = torch.mean(frames.float(), dim=(0, 2, 3), keepdim=True)
            std = torch.std(frames.float(), dim=(0, 2, 3), keepdim=True)
            return (frames - mean) / std
        
    def load_alignments(self,idx):
        tokens = []
        with open(self.align_list[idx], 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            if line and line[2] != 'sil':
                tokens.extend([' '] + list(line[2]))  # 공백을 추가하고 문자 단위로 분할하여 리스트에 추가

        # 문자열을 숫자로 변환
        num_tokens = char_to_num(tokens[1:])  # 첫 번째 공백을 제외하고 변환
        target_length = 40
        current_length = num_tokens.size(0)
        if current_length < target_length:
            # (앞쪽 패딩, 뒤쪽 패딩) 순서로 패딩을 추가
            pad_size = (0, target_length - current_length)  # 뒤쪽에만 패딩을 추가
            num_tokens = F.pad(num_tokens.view(1, -1), pad_size, "constant", 0).view(-1)  # 패딩 추가 후 원래 차원으로 복구
        return num_tokens

class SimpleLipnet(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.conv_layer=nn.Sequential(
            nn.Conv3d(in_channels=75, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(in_channels=256, out_channels=75, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.lstm_layer=nn.ModuleList([
            nn.LSTM(input_size=85, 
                            hidden_size=128, 
                            bidirectional=True, 
                            batch_first=True),
            nn.Dropout(p=0.5),
            nn.LSTM(input_size=128*2, 
                            hidden_size=128, 
                            bidirectional=True, 
                            batch_first=True),
            nn.Dropout(p=0.5)]
        )
        self.linear=nn.Linear(128*2,vocab_size+1)
        init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x=self.conv_layer(x)
        batch_size, seq_len, channels, height, width = x.size()
        x=x.view(batch_size, seq_len, -1)
        
        for layer in self.lstm_layer:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        x=F.log_softmax(self.linear(x), dim=-1)
        
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        return x
    
def char_to_num(characters):
    return torch.tensor([char_to_index[char] for char in characters])

def num_to_char(numbers):
    return ''.join([index_to_char[number] for number in numbers.tolist()])

def compute_ctc_loss(y_pred, y_true):
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    batch_size=y_pred.shape[1]
    input_length = torch.full((batch_size,), y_pred.size(0), dtype=torch.long)  # [T, N, C]에서 T는 시퀀스 길이
    target_length = torch.full((batch_size,), y_true.size(1), dtype=torch.long)  # y_true의 형태는 [N, S] 입니다.
    
    loss = ctc_loss(y_pred, y_true, input_length, target_length)
    return loss

def custom_ctc_decode(y_pred, blank_label=0):
    """
    간단한 그리디 CTC 디코딩 함수.
    
    Args:
    y_pred (torch.Tensor): 모델의 출력 텐서. 크기는 (batch_size, seq_len, num_classes)입니다.
    blank_label (int): "blank" 레이블의 인덱스.
    
    Returns:
    decoded_sequences (list of lists): 디코딩된 레이블 시퀀스의 리스트.
    """
    # 가장 가능성이 높은 인덱스를 선택합니다.
    y_pred = y_pred.permute(1, 0, 2)
    _, max_indices = torch.max(y_pred, 2)
    
    decoded_sequences = []
    for indices in max_indices:
        sequence = []
        last_idx = None
        for idx in indices:
            # 연속된 중복 제거 및 blank 레이블 제거
            if idx != last_idx and idx != blank_label:
                sequence.append(idx.item())
            last_idx = idx
        decoded_sequences.append(sequence)
    
    return decoded_sequences

class ProduceExample:
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader) 

    def on_epoch_end(self, epoch, model, num_to_char):
        data, targets = next(self.dataloader)
        data = data.to(model.device)  # 모델과 같은 디바이스로 데이터 이동
        yhat = model(data)
        
        # CTC 디코드 로직은 별도 구현 또는 라이브러리 사용 필요
        decoded = custom_ctc_decode(yhat)
        for i in range(len(yhat)):
            original_text = ''.join([num_to_char[idx] for idx in targets[i].tolist()])
            predicted_text = ''.join([num_to_char[idx] for idx in decoded[i].tolist()])
            
            print('Original:', original_text)
            print('Prediction:', predicted_text)
            print('~' * 100)

if __name__=="__main__":
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="simple-lipnet2", 
        config={"dropout": 0.1, "learning_rate": 1e-2},
    )
    device=accelerator.device
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"Using GPU indices: {accelerator.state.process_index}")
    
    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    vocab_size=len(vocab)
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {index: char for index, char in enumerate(vocab)}
    dataset=CustomDataset()
    train_size = 500
    test_size = 500
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,drop_last=True)
    model=SimpleLipnet(vocab_size).to(device)
    # model.to(device)  # 모델을 적절한 device로 이동
    optimizer = Adam(model.parameters())
    
    train_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, test_loader, model, optimizer
    )

    # Training loop (simplified version)
    num_epochs=100
    for epoch in range(num_epochs):
        total_loss=0
        num=0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(data)
            loss = compute_ctc_loss(predictions, targets)*100
            num+=1
            total_loss=loss.item()
            # Backward and optimize
            
            accelerator.backward(loss)
            # progress_bar.update(1)
            optimizer.step()
            # print(loss)
        accelerator.log({"training_loss": total_loss/len(train_loader)})
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
        
        save_path = f'/workspace/lipnet-test/checkpoints/model_epoch_{epoch+1}.pth'
        accelerator.save_state_dict(model.state_dict(), save_path)
    
    accelerator.end_training()
