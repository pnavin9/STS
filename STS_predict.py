import torch
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class STSBertModel(torch.nn.Module):
    def __init__(self):

        super(STSBertModel, self).__init__()

        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=128)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def forward(self, input_data):

        output = self.sts_model(input_data)
        
        return output
trained_model =  STSBertModel()
trained_model.load_state_dict(torch.load('artifacts/model.pt'))
def predict_sts(texts):
  #trained_model.to('cpu')
  #trained_model.eval()
  #print(trained_model.keys)
  test_input = tokenizer(texts, padding='max_length', max_length = 128, truncation=True, return_tensors="pt")
  test_input['input_ids'] = test_input['input_ids']
  test_input['attention_mask'] = test_input['attention_mask']
  del test_input['token_type_ids']

  test_output = trained_model(test_input)['sentence_embedding']
  sim = torch.nn.functional.cosine_similarity(test_output[0], test_output[1], dim=0).item()

  return sim
print(predict_sts(['My name is Navin','Navin is my name']))