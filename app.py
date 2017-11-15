"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Look for a number of word and the temperature
    - Returns the evaluation

POST req:
    parameter:
        - words, required, how many words to generate
        - temperature, optional, degree of diversity

"""
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
import torch
from torch.autograd import Variable
import data

DATA_PATH = '/input'
CHECKPOINT = '/model/model.pt'
OUTPUT_PATH = '/output/generated.txt'
LOG_INTERVAL = 50
print('Loading checkpoint: %s' % CHECKPOINT)

app = Flask('Language-Model-Text-Generator')

# Check if ckp exists
try:
    os.path.isfile(CHECKPOINT)
except IOError as e:
        # Does not exist OR no read permissions
    print ("Unable to open ckp file")

cuda = torch.cuda.is_available()

# Load checkpoint
if cuda:
    model = torch.load(CHECKPOINT)
else:
    # Load GPU model on CPU
    model = torch.load(CHECKPOINT, map_location=lambda storage, loc: storage)
model.eval()

if cuda:
    model.cuda()
else:
    model.cpu()

# Load Data
corpus = data.Corpus(DATA_PATH)
ntokens = len(corpus.dictionary)

def generate(words, temperature):
    """Generate number of words with the given temperature"""
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    if cuda:
        input.data = input.data.cuda()

    # Generate
    with open(OUTPUT_PATH, 'w') as outf:
        for i in range(words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            # word = '\n' if word == "<eos>" else word
            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % LOG_INTERVAL == 0:
                print('| Generated {}/{} words'.format(i, words))

# Return an Text Generated
@app.route('/<path:path>', methods=['POST'])
def geneator_handler(path):
    # Get ckp
    words = int(request.form.get("words"))
    if words is None:
        return BadRequest("You must provide a words parameter")
    # if words is not int:
    #     return BadRequest("Invalid words type")
    temp = request.form.get("temperature") or 1.0
    temp = float(temp)
    # if type(temp) is not float or type(temp) is not int:
    #     return BadRequest("Invalid temperature type")
    if temp < 1e-3:
        return BadRequest("Temperature has to be greater or equal 1e-3")
    print (words, temp)
    # Generate word
    generate(words, temp)
    # Return the text generated
    return send_file(OUTPUT_PATH, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
