# %%
# Python modules. 
import spacy 



# %%
def to_spacy_document(nlp:spacy.tokens.doc.Doc, data:tuple) -> spacy.tokens.doc.Doc: 
	'''
	Convert the TRAIN and TEST into SpaCy document. For more information on
	how to enable and disable SpaCy pipeline check out the following links: 
	- https://spacy.io/api/language#pipe 
	- https://spacy.io/usage/processing-pipelines
	- https://spacy.io/usage/processing-pipelines#disabling
	'''
	print("Reformatting the dataset...") 

	# To store a list of (Spacy Doc) object. 
	text = []

	# Use SpaCy Doc object to categorise the sentiment. 
	for doc, label in nlp.pipe(data, as_tuples=True): 
		if label =="positive":
			doc.cats["positive"] = 1
			doc.cats["negative"] = 0
			doc.cats["neutral"]  = 0
		elif label == "negative":
			doc.cats["positive"] = 0
			doc.cats["negative"] = 1
			doc.cats["neutral"]  = 0
		else:
			doc.cats["positive"] = 0
			doc.cats["negative"] = 0
			doc.cats["neutral"]  = 1
		text.append(doc) 

	print("Reformatted the dataset.") 

	return text 
