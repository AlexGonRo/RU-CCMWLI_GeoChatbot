from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from keras.callbacks import EarlyStopping


class Seq2seq:

	'''
	Constructor
	@params input_dim
	@params latent_dim
	@params timesteps
	'''
	def __init__(self, input_dim, latent_dim, timesteps, batch_size, word_index, embedding_matrix):

		self.batch_size = batch_size
		self.word_index = word_index
		self.embedding_matrix = embedding_matrix

		inputs = Input(shape=(timesteps, input_dim))
		encoded = LSTM(latent_dim)(inputs)

		decoded = RepeatVector(timesteps)(encoded)
		decoded = LSTM(input_dim, return_sequences=True)(decoded)

		self.sequence_autoencoder = Model(inputs, decoded)
		self.sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer='sgd')
		self.encoder = Model(inputs, encoded)
		self.encoder.compile(loss='categorical_crossentropy', optimizer='sgd')

	'''
	Train classifier with training data.
	All required parameters for training except for data is passed to the
	constructor.
	'''
	def fit(self, train, validation=None, epochs=10):
		#early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

		self.sequence_autoencoder.fit(train,train,batch_size=self.batch_size,
								epochs=epochs, verbose=1, validation_split = 0.1) # callbacks=[early]

	'''
	Predict class labels on test data
	'''
	def predict(self, test):
		return self.encoder.predict(test, verbose=1)


	'''
	Evaluate classifier performance of validation data
	'''
	def evaluate(self, validation):
		score = self.sequence_autoencoder.evaluate(validation,validation,batch_size=self. batch_size)
		return score[0]