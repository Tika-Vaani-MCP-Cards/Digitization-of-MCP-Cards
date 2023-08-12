import os
import time
import string
import argparse
import re
import shutil
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataloader import MCPDatesDataset
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, criterion, test_loader, converter, opt, full_log=None):
	""" validation or evaluation """
	n_correct = 0
	norm_ED = 0
	length_of_data = 0
	infer_time = 0
	valid_loss_avg = Averager()

	for i, (image_tensors, labels, img_abs_path) in enumerate(tqdm(test_loader)):
		batch_size = image_tensors.size(0)
		length_of_data = length_of_data + batch_size
		image = image_tensors.to(device)
		# For max length prediction
		length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
		text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

		text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

		start_time = time.time()
		if 'CTC' in opt.Prediction:
			preds = model(image, text_for_pred)
			forward_time = time.time() - start_time

			# Calculate evaluation loss for CTC deocder.
			preds_size = torch.IntTensor([preds.size(1)] * batch_size)
			# permute 'preds' to use CTCloss format
			if opt.baiduCTC:
				cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
			else:
				cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

			# Select max probabilty (greedy decoding) then decode index to character
			if opt.baiduCTC:
				_, preds_index = preds.max(2)
				preds_index = preds_index.view(-1)
			else:
				_, preds_index = preds.max(2)
			preds_str = converter.decode(preds_index.data, preds_size.data)
		
		else:
			preds = model(image, text_for_pred, is_train=False)
			forward_time = time.time() - start_time

			preds = preds[:, :text_for_loss.shape[1] - 1, :]
			target = text_for_loss[:, 1:]  # without [GO] Symbol
			cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

			# select max probabilty (greedy decoding) then decode index to character
			_, preds_index = preds.max(2)
			preds_str = converter.decode(preds_index, length_for_pred)
			labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

		infer_time += forward_time
		valid_loss_avg.add(cost)

		# calculate accuracy & confidence score
		preds_prob = F.softmax(preds, dim=2)
		preds_max_prob, _ = preds_prob.max(dim=2)
		confidence_score_list = []
		for gt, pred, pred_max_prob, img_path in zip(labels, preds_str, preds_max_prob, img_abs_path):
			if 'Attn' in opt.Prediction:
				gt = gt[:gt.find('[s]')]
				pred_EOS = pred.find('[s]')
				pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
				pred_max_prob = pred_max_prob[:pred_EOS]

			# To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
			if opt.sensitive and opt.data_filtering_off:
				pred = pred.lower()
				gt = gt.lower()
				alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
				out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
				pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
				gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

			if pred == gt:
				n_correct += 1

			'''
			(old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
			"For each word we calculate the normalized edit distance to the length of the ground truth transcription."
			if len(gt) == 0:
				norm_ED += 1
			else:
				norm_ED += edit_distance(pred, gt) / len(gt)
			'''

			# ICDAR2019 Normalized Edit Distance
			if len(gt) == 0 or len(pred) == 0:
				norm_ED += 0
			elif len(gt) > len(pred):
				norm_ED += 1 - edit_distance(pred, gt) / len(gt)
			else:
				norm_ED += 1 - edit_distance(pred, gt) / len(pred)

			# calculate confidence score (= multiply of pred_max_prob)
			try:
				confidence_score = pred_max_prob.cumprod(dim=0)[-1]
				charwise_confidence = pred_max_prob.tolist()
				if full_log:
					if pred == gt:
						full_log.write("{} {},{},{},{}_{}\n".format(1, gt, pred, img_path, confidence_score, charwise_confidence))
					else:
						full_log.write("{} {},{},{},{}_{}\n".format(0, gt, pred, img_path, confidence_score, charwise_confidence))
			except:
				confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
			
			confidence_score_list.append(confidence_score)
			# print(pred, gt, pred==gt, confidence_score)

	accuracy = (n_correct / float(length_of_data)) * 100
	norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
	print('WRR- ', accuracy)
	print('CRR- ',100*norm_ED)
	return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def test(opt):
	""" model configuration """
	if 'CTC' in opt.Prediction:
		converter = CTCLabelConverter(opt.character)
	else:
		converter = AttnLabelConverter(opt.character)
	opt.num_class = len(converter.character)

	if opt.rgb:
		opt.input_channel = 3
	model = Model(opt)
	print('model input parameters', opt.img_h, opt.img_w, opt.num_fiducial, opt.input_channel, opt.output_channel,
		  opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
		  opt.SequenceModeling, opt.Prediction)
	model = torch.nn.DataParallel(model).to(device)

	# load model
	print('loading pretrained model from %s' % opt.saved_model)
	model.load_state_dict(torch.load(opt.saved_model, map_location=device))
	opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
	# print(model)

	""" keep evaluation model and result logs """
	os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
	# os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

	""" setup loss """
	if 'CTC' in opt.Prediction:
		criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
	else:
		criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

	""" evaluation """
	model.eval()
	with torch.no_grad():  
		log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
		# write prediction, gt, image-key, confidence score to file
		full_log = open(f'./result/{opt.exp_name}/full_log_evaluation.txt', 'w')

		eval_dataset = MCPDatesDataset(opt.eval_data, opt, is_training=False, data_aug=False, img_w=opt.img_w, img_h=opt.img_h)
		
		test_loader = torch.utils.data.DataLoader(
			eval_dataset, batch_size=opt.batch_size,
			shuffle=True,  # 'True' to check training progress with validation function.
			num_workers=int(opt.workers),
			pin_memory=True)

		_, accuracy_by_best_model, _, _, _, _, _, _ = validation(
			model, criterion, test_loader, converter, opt, full_log)
			
		print(f'{accuracy_by_best_model:0.3f}')
		log.write(f'{accuracy_by_best_model:0.3f}\n')
		log.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
	parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
	""" Data processing """
	parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
	parser.add_argument('--img_h', type=int, default=32, help='the height of the input image')
	parser.add_argument('--img_w', type=int, default=144, help='the width of the input image')
	parser.add_argument('--rgb', action='store_true', help='use rgb input')
	parser.add_argument('--character', type=str, default='0123456789/', help='character label')
	parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
	parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
	parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
	parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
	""" Model Architecture """
	parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
	parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
	parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
	parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
	parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
	parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
	parser.add_argument('--output_channel', type=int, default=512,
						help='the number of output channel of Feature extractor')
	parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
	parser.add_argument('--output_file', type=str, default=None, help='output prediction_confidence file name')
	opt = parser.parse_args()
	opt.training = False
	""" vocab / character number configuration """
	if opt.sensitive:
		opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
	opt.character = '0123456789/'
	cudnn.benchmark = True
	cudnn.deterministic = True
	opt.num_gpu = torch.cuda.device_count()

	test(opt)
	# write a better name for this variable opt.pred_confi_file_name

	if opt.output_file is not None:
		os.makedirs('TextResults', exist_ok=True)
		shutil.copy(f'./result/{opt.exp_name}/full_log_evaluation.txt', f'TextResults/{opt.output_file}')
