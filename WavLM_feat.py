# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Bogdan Vlasenko <bogdan.vlasenko@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor
from transformers import WavLMModel

parser = argparse.ArgumentParser(
    description="Some description of the script",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("input", type=str, help="Path to input wav file")
parser.add_argument("label", type=str, help="Label")
parser.add_argument("output", type=str, help="Path to output feature file (csv format)")
args = parser.parse_args()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = WavLMModel.from_pretrained("microsoft/wavlm-large")

speech, rate = librosa.load(args.input, sr=16000)
inputs = processor(speech, sampling_rate=rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
wavlm = last_hidden_states.cpu().detach().numpy()

mean = np.mean(wavlm[0], axis=0)
std = np.std(wavlm[0], axis=0)
lab = np.array([args.label])
out = np.concatenate(
    (mean.reshape(1, -1), std.reshape(1, -1), lab.reshape(1, -1)), axis=1
)
out_df = pd.DataFrame(out)
out_df.to_csv(args.output, mode="w", sep=",", header=None, index=False)
