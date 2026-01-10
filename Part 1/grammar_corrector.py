import tqdm as notebook_tqdm
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

# model_name = "prithivida/grammar_error_correcter_v1"
# gc1= pipeline("text2text-generation", model_name)


encoder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return encoder.encode(text)

def sematic_similarity(v1,v2):
    embedv1 = embed(v1)
    embedv2 = embed(v2)
    return np.dot(embedv1,embedv2)/(np.linalg.norm(embedv1)*np.linalg.norm(embedv2))

class GrammarCorrector:
    def __init__(
        self,
        model_name="prithivida/grammar_error_correcter_v1",
        max_length = 256,
        use_prompt = False
    ):
        self.model_name = model_name
        self.max_length = max_length
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_prompt = use_prompt

        self.pipe = pipeline(
            "text2text-generation",
            model=self.model_name,
            device= device
        )

    def _correct_once(self, text) -> str:
        text = self.prepare(text)
        """Run the model one time."""
        out = self.pipe(
        text,
        max_new_tokens=256,                 #Limits the number of tokens the model can generate in addition to the input
        truncation=False,                   #tuncation = false throws error if input exeeds models maximun input length
        clean_up_tokenization_spaces=True,  #Removes exrta spaces 
        do_sample = False,                  #Deterministic output and not creative based of sampled tokens 
        num_beams = 4,                      #number of alternative outputs the model creates. 
        repetition_penalty = 1.3,           #penalty for repeating the same token 
        )[0]["generated_text"]

        return out.strip()

    def prepare(self,text) -> str:
        '''Prepares text by adding prompt if required'''
        if self.use_prompt:
            return f"Rewrite the following text to be grammatically correct, concise, and without repetition.Do not add new information:\n{text}"
        return text

    def split_text(self,text: str) -> list[str]:
        words = text.split()
        chunks = []
        current= ""

        for word in words:
            if len(current) + len(word) + 1 <self.max_length:
                current += word + " "
            else:
                chunks.append(current.strip())
                current= word + " "
        
        if current:
            chunks.append(current.strip())
        return chunks


    def correct(self, text:str) -> tuple[str,float]:
        '''
        Correct the full text:        
        -   Split into chunks if text is too big
        -   Correct each chuck indepentently
        -   Returns corrected text and sematic similarity score
        '''
        if len(text) <= self.max_length:
            chunks =[text]
        else:
            chunks = self.split_text(text)
        
        corrected_chunks = []
        for chunk in chunks:
            current= chunk
            candidate = self._correct_once(current)
            current = candidate

            corrected_chunks.append(current)
            

        final =" ".join(corrected_chunks)
        score = sematic_similarity(text, final)
        return final,score