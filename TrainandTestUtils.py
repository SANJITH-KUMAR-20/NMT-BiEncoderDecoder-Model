import Model as mm
import torch
import os
import Utils as ut
import random
import matplotlib.pyplot as plt
import psutil
import math
import time

def train_batch(input_batch, target_batch, encoder, decoder, 
                encoder_optimizer, decoder_optimizer, loss_criterion,output_lang,clip=1.0):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0
	enc_h_hidden, enc_c_hidden = encoder.initialize_weights(input_batch.shape[1])

	enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)

	decoder_input = torch.autograd.Variable(torch.LongTensor(1,input_batch.shape[1]).
                           fill_(output_lang.word_to_idx.get("SOS")).cuda()) if torch.cuda.is_available() \
					else torch.autograd.Variable(torch.LongTensor(1,input_batch.shape[1]).
                        fill_(output_lang.word_to_idx.get("SOS")))

	dec_h_hidden = enc_outputs[0]
	dec_c_hidden = enc_outputs[1]
	
	for i in range(target_batch.shape[0]):
		pred, dec_outputs = decoder(decoder_input, dec_h_hidden, 
                                dec_c_hidden, enc_hiddens)

		decoder_input = target_batch[i].view(1,-1)
		dec_h_hidden = dec_outputs[0]
		dec_c_hidden = dec_outputs[1]
		
		loss += loss_criterion(pred,target_batch[i])


	loss.backward()

	torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
	torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_batch.shape[0]

def train(train_batches, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion,output_lang):

	round_loss = 0
	i = 1
	for batch in train_batches:
		i += 1
		(input_batch, target_batch) = ut.pad_batch(batch)
		batch_loss = train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion,output_lang)
		round_loss += batch_loss

	return round_loss / len(train_batches)

def test_batch(input_batch, target_batch, encoder, decoder, loss_criterion,output_lang):
	
	loss = 0

	enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])

	enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)

	decoder_input = torch.autograd.Variable(torch.LongTensor(1,input_batch.shape[1]).
                           fill_(output_lang.word_to_index.get("SOS")).cuda()) if torch.cuda.is_available() \
					else torch.autograd.Variable(torch.LongTensor(1,input_batch.shape[1]).
                        fill_(output_lang.word_to_index.get("SOS")))
	dec_h_hidden = enc_outputs[0]
	dec_c_hidden = enc_outputs[1]
	
	for i in range(target_batch.shape[0]):
		pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

		topv, topi = pred.topk(1,dim=1)
		ni = topi.view(1,-1)
		
		decoder_input = ni
		dec_h_hidden = dec_outputs[0]
		dec_c_hidden = dec_outputs[1]

		loss += loss_criterion(pred,target_batch[i])
		
	return loss.item() / target_batch.shape[0]

def test(test_batches, encoder, decoder, loss_criterion,output_lang):

	with torch.no_grad():
		test_loss = 0

		for batch in test_batches:
			(input_batch, target_batch) = ut.pad_batch(batch)
			batch_loss = test_batch(input_batch, target_batch, encoder, decoder, loss_criterion,output_lang)
			test_loss += batch_loss

	return test_loss / len(test_batches)

def evaluate(encoder, decoder, sentence, cutoff_length,input_lang,output_lang):
	with torch.no_grad():
		input_variable = ut.tensorFromSentence(input_lang, sentence)
		input_variable = input_variable.view(-1,1)
		enc_h_hidden, enc_c_hidden = encoder.initialize_weights(1)

		enc_hiddens, enc_outputs = encoder(input_variable, enc_h_hidden, enc_c_hidden)

		decoder_input = torch.autograd.Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_idx.get("SOS")).cuda()) if torch.cuda.is_available() \
						else torch.autograd.Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_idx.get("SOS")))
		dec_h_hidden = enc_outputs[0]
		dec_c_hidden = enc_outputs[1]

		decoded_words = []

		for di in range(cutoff_length):
			pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

			topv, topi = pred.topk(1,dim=1)
			ni = topi.item()
			if ni == output_lang.word_to_idx.get("EOS"):
				decoded_words.append('')
				break
			else:
				decoded_words.append(output_lang.idx_to_word[ni])

			decoder_input = torch.autograd.Variable(torch.LongTensor(1,1).fill_(ni).cuda()) if torch.cuda.is_available() \
							else torch.autograd.Variable(torch.LongTensor(1,1).fill_(ni))
			dec_h_hidden = dec_outputs[0]
			dec_c_hidden = dec_outputs[1]

		output_sentence = ' '.join(decoded_words)
		return output_sentence
	
def evaluate_randomly(encoder, decoder, pairs,input_lang,output_lang, n=2, trim=100):
	for i in range(n):
		pair = random.choice(pairs)
		print('>', pair[0])
		print('=', pair[1])
		output_sentence = evaluate(encoder, decoder, pair[0],trim,input_lang,output_lang)
		print('<', output_sentence)
		print('')    
		
'''Used to plot the progress of training. Plots the loss value vs. time'''
def showPlot(times, losses, fig_name):
    x_axis_label = 'Minutes'
    colors = ('red','blue')
    if max(times) >= 120:
        times = [mins/60 for mins in times]
    x_axis_label = "Hourse"
    i = 0
    for key, losses in losses.items():
        if len(losses) > 0:
            plt.plot(times, losses, label=key, color=colors[i])
            i += 1
    plt.legend(loc='upper left')
    plt.xlabel(x_axis_label)
    plt.ylabel('Loss')
    plt.title('Training Results')
    plt.savefig(fig_name+'.png')
    plt.close('all')
    
'''prints the current memory consumption'''
def mem():
	if torch.cuda.is_available():
		mem = torch.cuda.memory_allocated()/1e7
	else:
		mem = psutil.cpu_percent()
	print('Current mem usage:')
	print(mem)
	return "Current mem usage: %s \n" % (mem)

def asHours(s):
	m = math.floor(s / 60)
	h = math.floor(m / 60)
	s -= m * 60
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)

def train_and_test(epochs, test_eval_every, plot_every, learning_rate, 
                   lr_schedule, train_pairs, test_pairs, input_lang, 
                   output_lang, batch_size, test_batch_size, encoder, decoder, 
                   loss_criterion, trim, save_weights,output_file_name):
	
	times = []
	losses = {'train set':[], 'test set': []}

	test_batches, longest_seq, n_o_b = ut.batchify(test_pairs, input_lang, 
                                              output_lang, test_batch_size, 
                                              shuffle_data=False)

	start = time.time()
	for i in range(1,epochs+1):
    
		'''adjust the learning rate according to the learning rate schedule
		specified in lr_schedule'''
		if i in lr_schedule.keys():
			learning_rate /= lr_schedule.get(i)


		encoder.train()
		decoder.train()

		encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
		decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

		batches, longest_seq, n_o_b = ut.batchify(train_pairs, input_lang, 
                                           output_lang, batch_size, 
                                           shuffle_data=True)
		train_loss = train(batches, encoder, decoder, encoder_optimizer, 
                       decoder_optimizer, loss_criterion,output_lang)
		now = time.time()
		print("Iter: %s \nLearning Rate: %s \nTime: %s \nTrain Loss: %s \n" 
          % (i, learning_rate, asHours(now-start), train_loss))

		if i % test_eval_every == 0:
			if test_pairs:
				test_loss = test(test_batches, encoder, decoder, loss_criterion,output_lang)
				print("Test set loss: %s" % (test_loss))
				evaluate_randomly(encoder, decoder,input_lang,output_lang,test_pairs,trim)
			else:
				evaluate_randomly(encoder, decoder,input_lang,output_lang, train_pairs, trim)

		if i % plot_every == 0:
			times.append((time.time()-start)/60)
			losses['train set'].append(train_loss)
			if test_pairs:
				losses['test set'].append(test_loss)
			showPlot(times, losses, output_file_name)
			if save_weights:
				torch.save(encoder.state_dict(), output_file_name+'_enc_weights.pt')
				torch.save(decoder.state_dict(), output_file_name+'_dec_weights.pt')

