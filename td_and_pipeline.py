import pandas as pd

#from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
# using pipeline API for summarization task
summarization = pipeline("summarization")

def summary(text):
    """
    Extracting text summary using transformers
    """
    try:
        summary_text = summarization(text)[0]['summary_text']
    except IndexError:
        # Split the original text into smaller chunks
        chunk_size = 512
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # Generate the summary for each chunk
        summary_text = ""
        for chunk in text_chunks:
            summary_text += summarization(chunk)[0]['summary_text']
    
    return summary_text


def classify(text):
    return analyzer.polarity_scores(text)

def calculate_chunk_size(text, max_chunk_size):
    # Get the length of the text
    text_length = len(text)
    
    # Calculate the number of chunks needed
    num_chunks = (text_length + max_chunk_size - 1) // max_chunk_size
    
    # Calculate the actual chunk size
    actual_chunk_size = text_length // num_chunks
    
    return actual_chunk_size


def vadersentimentanalysis(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def t5_summary(text):
    """
    Extracting text summary using t5 
    """

    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="tf", truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary tokens back to text
    summary_text = tokenizer.decode(summary_ids.numpy()[0], skip_special_tokens=True)
    
    return summary_text


df = pd.read_excel('news_summary.xlsx')
df.drop_duplicates(subset='Title',inplace=True)

# Load pre-trained T5 model and tokenizer
model = TFT5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")


#%%
analyzer = SentimentIntensityAnalyzer()

df['Pol_score'] = df["Summary"].apply(vadersentimentanalysis)

# max(zip(vs.values(), vs.keys()))

df['classification'] = df['Summary'].apply(classify)

#%%

# Example usage:
text = "india initiate anti dumping probe import solar glass china vietnam  India has initiated an anti dumping probe into imports of certain solar glass from China and Vietnam  following a complaint by domestic players  The commerce ministry s investigation arm Directorate General of Trade Remedies  DGTR  is probing the alleged dumping of  Textured Tempered Coated and Uncoated Glass  made or from China and Vietnam  The product is also known by various names such as solar glass or solar photovoltaic glass in the market parlance  An application has been filed by Borosil Renewables Ltd on behalf of the domestic industry for the probe and the imposition of appropriate anti dumping duty on imports   On the basis of the duly substantiated application by the domestic industry  and having satisfied itself  on the basis of prima facie evidence submitted by the applicant substantiating the dumping and consequent injury to the domestic industry  the authority hereby initiates an anti dumping investigation into the alleged dumping   the notification said  If it is established that the dumping has caused material injury to domestic players  DGTR would recommend the imposition of anti dumping duty on the imports  The finance ministry takes the final decision to impose duties  There is sufficient evidence that the product is being dumped in the domestic market of India by the exporters from these two countries  Anti dumping probes are conducted by countries to determine whether domestic industries have been hurt because of a surge in cheap imports  As a countermeasure  they impose these duties under the multilateral regime of the Geneva based World Trade Organisation  WTO   The duty is aimed at ensuring fair trading practices and creating a level playing field for domestic producers vis a vis foreign producers and exporters  India has already imposed anti dumping duty on several products to tackle cheap imports from various countries  including China "

summary_text = summary(text)

vs = analyzer.polarity_scores(" An application has been filed by Borosil Renewables Ltd on behalf of the domestic industry for the probe and the imposition of appropriate anti-dumping duty on imports.")

max_chunk_size = 512

chunk_size = calculate_chunk_size(text, max_chunk_size)
print("Chunk size:", chunk_size)

#%%
